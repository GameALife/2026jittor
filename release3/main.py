import os
import os.path as osp
import sys
import json

# Prefer local JittorGeometric (patched CRAFT) over site-packages
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, root)
sys.path.insert(0, osp.join(root, 'JittorGeometric'))

# Default async GPU; set JT_SYNC=1 only when debugging numerical issues
os.environ.setdefault('JT_SYNC', '0')

import jittor as jt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from jittor_geometric.data import TemporalData
from jittor_geometric.nn.models.craft import CRAFT
from jittor_geometric.dataloader.temporal_dataloader import TemporalDataLoader, get_neighbor_sampler
from jittor_geometric.evaluate.evaluators import MRR_Evaluator
import argparse

from config import get_dataset_config
from fast_ops import (
    build_neighbor_csr, sample_recent_neighbors_left, last_update_times,
    warmup_fast_ops, _HAS_NUMBA,
)

# Line-buffered stdout so epoch logs appear immediately under `tee` / pipes
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

jt.flags.use_cuda = 1


def log(msg=''):
    """Print and flush (visible immediately when stdout is piped to tee)."""
    print(msg, flush=True)


def build_craft_model(cfg, node_size):
    """Construct CRAFT from a dataset config dict."""
    return CRAFT(
        n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'],
        hidden_size=cfg['hidden_size'],
        hidden_dropout_prob=cfg['hidden_dropout_prob'],
        attn_dropout_prob=cfg['attn_dropout_prob'],
        hidden_act='gelu',
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        n_nodes=node_size,
        max_seq_length=cfg['num_neighbors'],
        loss_type=cfg['loss_type'],
        use_pos=cfg['use_pos'],
        input_cat_time_intervals=cfg['input_cat_time_intervals'],
        output_cat_time_intervals=cfg['output_cat_time_intervals'],
        output_cat_repeat_times=cfg['output_cat_repeat_times'],
        num_output_layer=1,
        emb_dropout_prob=cfg['emb_dropout_prob'],
        skip_connection=cfg['skip_connection'],
    )


def reshape_neg_dst(neg_dst, batch_size, num_neg):
    """Reshape flat loader negatives to [B, K]."""
    neg = np.asarray(neg_dst).reshape(-1)
    if neg.size != batch_size * num_neg:
        # Fallback: truncate/pad if loader size mismatches
        need = batch_size * num_neg
        if neg.size < need:
            pad = np.random.randint(int(neg.min()), int(neg.max()) + 1, size=need - neg.size)
            neg = np.concatenate([neg, pad])
        neg = neg[:need]
    return neg.reshape(batch_size, num_neg).astype(np.int32)


def avoid_pos_collision(pos_dst, neg_dst, min_dst, max_dst, rng, max_tries=10):
    """Resample negatives that collide with the positive dst of the same row."""
    neg = neg_dst.copy()
    pos = pos_dst.reshape(-1, 1)
    for _ in range(max_tries):
        mask = neg == pos
        if not mask.any():
            break
        neg[mask] = rng.randint(min_dst, max_dst + 1, size=int(mask.sum()))
    return neg


def inject_hard_negatives(pos_dst, neg_dst, src_neighb_seq, hard_neg_ratio, rng,
                          pop_ids=None, pop_probs=None, pop_hard_ratio=0.0):
    """Replace a fraction of random negatives with history and/or popular dsts.

    History path is vectorized (fixed slots) for speed on large batches.
    """
    hist_r = float(hard_neg_ratio or 0.0)
    pop_r = float(pop_hard_ratio or 0.0)
    if hist_r <= 0 and pop_r <= 0:
        return neg_dst
    neg = neg_dst.copy()
    bs, k = neg.shape
    n_hist = min(k, max(0, int(round(k * hist_r))))
    n_pop = min(k - n_hist, max(0, int(round(k * pop_r))))
    hist = np.asarray(src_neighb_seq) if src_neighb_seq is not None else None
    pos = np.asarray(pos_dst).reshape(-1)

    # Fast history hard-negs: always overwrite first n_hist columns (order doesn't matter)
    if hist is not None and n_hist > 0:
        # Pick a random column from history as candidate; fall back keeps random neg
        L = hist.shape[1]
        pick = rng.randint(0, L, size=(bs, n_hist))
        row_idx = np.arange(bs)[:, None]
        cand = hist[row_idx, pick]
        # invalidate zeros / positive collisions → keep original random neg there
        bad = (cand == 0) | (cand == pos.reshape(-1, 1))
        if bad.any():
            # one retry with another random history column
            pick2 = rng.randint(0, L, size=(bs, n_hist))
            cand2 = hist[row_idx, pick2]
            use2 = bad & (cand2 != 0) & (cand2 != pos.reshape(-1, 1))
            cand = np.where(use2, cand2, cand)
            bad = (cand == 0) | (cand == pos.reshape(-1, 1))
        neg[:, :n_hist] = np.where(bad, neg[:, :n_hist], cand)

    if pop_ids is not None and pop_probs is not None and n_pop > 0:
        # Popular path still per-row (usually disabled for ds2)
        for i in range(bs):
            slots = np.arange(n_hist, n_hist + n_pop)
            chosen = rng.choice(pop_ids, size=n_pop, replace=True, p=pop_probs)
            for j, c in zip(slots, chosen):
                if int(c) == int(pos[i]) and len(pop_ids) > 1:
                    c = int(rng.choice(pop_ids))
                neg[i, j] = c
    return neg


def build_popular_dst_table(dst_np, top_k=5000):
    """Return (ids, probs) for sampling popular destinations by frequency."""
    dst_np = np.asarray(dst_np, dtype=np.int64).ravel()
    counts = np.bincount(dst_np)
    if counts.size == 0:
        return None, None
    top_k = min(int(top_k), int((counts > 0).sum()))
    if top_k <= 0:
        return None, None
    # take top-k by count
    ids = np.argpartition(counts, -top_k)[-top_k:]
    ids = ids[counts[ids] > 0]
    w = counts[ids].astype(np.float64)
    w = w / w.sum()
    return ids.astype(np.int32), w


def prepare_batch_dst(dst, neg_dst, num_neg, min_dst, max_dst, rng,
                      src_neighb_seq=None, hard_neg_ratio=0.0,
                      pop_ids=None, pop_probs=None, pop_hard_ratio=0.0):
    """Build test_dst = [pos | negs] with collision avoidance (+ optional hard negs)."""
    pos = np.asarray(dst, dtype=np.int32).reshape(-1)
    bs = len(pos)
    neg = reshape_neg_dst(neg_dst, bs, num_neg)
    if (hard_neg_ratio and hard_neg_ratio > 0) or (pop_hard_ratio and pop_hard_ratio > 0):
        neg = inject_hard_negatives(
            pos, neg, src_neighb_seq, hard_neg_ratio, rng,
            pop_ids=pop_ids, pop_probs=pop_probs, pop_hard_ratio=pop_hard_ratio)
    neg = avoid_pos_collision(pos, neg, min_dst, max_dst, rng)
    return np.concatenate([pos.reshape(bs, 1), neg], axis=1)


def get_dst_last_update_times(full_neighbor_sampler, test_dst_np, t_np, csr=None):
    """Last interaction time before t for each candidate."""
    if csr is not None:
        out_t = last_update_times(test_dst_np, t_np, csr)
        return jt.Var(out_t)
    bs, n_cand = test_dst_np.shape
    node_ids = test_dst_np.reshape(-1).astype(np.int64)
    times = np.broadcast_to(t_np[:, np.newaxis], (bs, n_cand)).reshape(-1).astype(np.float64)
    out_t = np.full(len(node_ids), -100000.0, dtype=np.float32)
    neigh_times = full_neighbor_sampler.nodes_neighbor_times
    for i, (nid, t) in enumerate(zip(node_ids, times)):
        arr = neigh_times[int(nid)]
        if arr.size == 0:
            continue
        idx = np.searchsorted(arr, t)
        if idx > 0:
            out_t[i] = arr[idx - 1]
    return jt.Var(out_t.reshape(bs, n_cand))


def prepare_train_features(batch_data, full_neighbor_sampler, num_neighbors, num_neg_train,
                           min_dst, max_dst, rng, hard_neg_ratio,
                           pop_ids=None, pop_probs=None, pop_hard_ratio=0.0, csr=None):
    """CPU-side neighbor sampling + last-update for one training batch."""
    src_np = np.asarray(batch_data.src.numpy() if hasattr(batch_data.src, 'numpy') else batch_data.src, dtype=np.int32)
    dst_np = np.asarray(batch_data.dst.numpy() if hasattr(batch_data.dst, 'numpy') else batch_data.dst, dtype=np.int32)
    t_np = np.asarray(batch_data.t.numpy() if hasattr(batch_data.t, 'numpy') else batch_data.t, dtype=np.int32)
    if csr is not None:
        src_neighb_seq, src_neighb_interact_times = sample_recent_neighbors_left(
            src_np, t_np, csr, num_neighbors)
    else:
        src_neighb_seq, _, src_neighb_interact_times = full_neighbor_sampler.get_historical_neighbors_left(
            node_ids=src_np, node_interact_times=t_np, num_neighbors=num_neighbors)
    neighbor_num = (src_neighb_seq != 0).sum(axis=1)
    if neighbor_num.sum() == 0:
        return None
    test_dst_np = prepare_batch_dst(
        dst_np, batch_data.neg_dst, num_neg_train, min_dst, max_dst, rng,
        src_neighb_seq=src_neighb_seq, hard_neg_ratio=hard_neg_ratio,
        pop_ids=pop_ids, pop_probs=pop_probs, pop_hard_ratio=pop_hard_ratio)
    dst_last_np = last_update_times(test_dst_np, t_np, csr) if csr is not None else None
    return {
        't_np': t_np,
        'src_neighb_seq': src_neighb_seq,
        'src_neighb_interact_times': src_neighb_interact_times,
        'neighbor_num': neighbor_num,
        'test_dst_np': test_dst_np,
        'dst_last_np': dst_last_np,
    }


def multi_neg_bpr_loss(pos_score, neg_score):
    """BPR over K negatives: mean -log σ(pos - neg_k). pos [B], neg [B, K] or flat."""
    pos = pos_score.reshape(-1, 1)
    if neg_score.ndim == 1:
        bs = pos.shape[0]
        neg = neg_score.reshape(bs, -1)
    else:
        neg = neg_score
    loss = -jt.log(1e-10 + jt.sigmoid(pos - neg))
    return loss.mean()


def sampled_softmax_loss(pos_score, neg_score, temperature=1.0):
    """Sampled softmax CE: treat pos as class 0 among [pos | K negs]. Aligns with MRR."""
    pos = pos_score.reshape(-1, 1)
    if neg_score.ndim == 1:
        neg = neg_score.reshape(pos.shape[0], -1)
    else:
        neg = neg_score
    logits = jt.concat([pos, neg], dim=1)  # [B, 1+K]
    temp = float(temperature or 1.0)
    if temp != 1.0:
        logits = logits / temp
    log_probs = jt.nn.log_softmax(logits, dim=-1)
    return (-log_probs[:, 0]).mean()


def compute_train_loss(pos_score, neg_score, train_loss, train_softmax_temp=1.0):
    """Dispatch training loss by config name."""
    name = (train_loss or 'bpr').lower()
    if name == 'softmax':
        return sampled_softmax_loss(pos_score, neg_score, temperature=train_softmax_temp)
    if name == 'bpr':
        return multi_neg_bpr_loss(pos_score, neg_score)
    raise ValueError(f'Unknown train_loss={train_loss!r}; use "bpr" or "softmax"')


def build_dst_log_popularity(dst_np, node_size):
    """log(1 + count) popularity vector indexed by node id."""
    counts = np.bincount(np.asarray(dst_np, dtype=np.int64).ravel(), minlength=node_size)
    return np.log1p(counts.astype(np.float64))


def build_src_dst_cooccur(src_np, dst_np):
    """Map src -> {dst: count} from training edges (for inference boost)."""
    from collections import defaultdict
    co = defaultdict(dict)
    src_np = np.asarray(src_np, dtype=np.int64).ravel()
    dst_np = np.asarray(dst_np, dtype=np.int64).ravel()
    for s, d in zip(src_np, dst_np):
        bucket = co[int(s)]
        bucket[int(d)] = bucket.get(int(d), 0) + 1
    return co


def history_boost_logits(logits, candidates, src_neighb_seq, src_neighb_times, cur_times, cfg,
                         batch_src=None, cooccur=None):
    """Boost logits using history hit / count / recency / train co-occurrence."""
    base = float(cfg.get('history_boost', 0.0) or 0.0)
    count_coef = float(cfg.get('history_count_coef', 0.0) or 0.0)
    recency_coef = float(cfg.get('history_recency_coef', 0.0) or 0.0)
    co_coef = float(cfg.get('cooccur_boost', 0.0) or 0.0)
    if base <= 0 and count_coef <= 0 and recency_coef <= 0 and co_coef <= 0:
        return logits

    out = np.asarray(logits, dtype=np.float64).copy()
    cand = np.asarray(candidates)
    hist = np.asarray(src_neighb_seq)
    hist_t = None if src_neighb_times is None else np.asarray(src_neighb_times)
    cur = None if cur_times is None else np.asarray(cur_times).reshape(-1)
    srcs = None if batch_src is None else np.asarray(batch_src).reshape(-1)

    for i in range(out.shape[0]):
        row = hist[i]
        valid = row != 0
        if not valid.any() and co_coef <= 0:
            continue
        # count + last time per neighbor id
        counts = {}
        last_t = {}
        for k, nid in enumerate(row):
            nid = int(nid)
            if nid == 0:
                continue
            counts[nid] = counts.get(nid, 0) + 1
            if hist_t is not None:
                tt = float(hist_t[i, k])
                if nid not in last_t or tt > last_t[nid]:
                    last_t[nid] = tt
        src_co = None
        if co_coef > 0 and cooccur is not None and srcs is not None:
            src_co = cooccur.get(int(srcs[i]), {})

        for j, c in enumerate(cand[i]):
            cid = int(c)
            add = 0.0
            cnt = counts.get(cid, 0)
            if cnt > 0:
                add += base
                if count_coef > 0:
                    add += count_coef * np.log1p(cnt)
                if recency_coef > 0 and cur is not None and cid in last_t:
                    delta = max(cur[i] - last_t[cid], 0.0)
                    add += recency_coef / (1.0 + np.log1p(delta))
            if src_co is not None:
                cc = src_co.get(cid, 0)
                if cc > 0:
                    add += co_coef * np.log1p(cc)
            if add:
                out[i, j] += add
    return out


def pop_boost_logits(logits, candidates, pop_prior, alpha):
    """Add alpha * row-minmax(log-pop) to logits (keeps ranking meaningful)."""
    if alpha is None or alpha <= 0 or pop_prior is None:
        return logits
    out = np.asarray(logits, dtype=np.float64).copy()
    pop = pop_prior[np.asarray(candidates, dtype=np.int64)]
    pop_min = pop.min(axis=1, keepdims=True)
    pop_max = pop.max(axis=1, keepdims=True)
    pop_n = (pop - pop_min) / (pop_max - pop_min + 1e-8)
    out = out + float(alpha) * pop_n
    return out


def logits_to_probs(logits, normalize='softmax', temperature=1.0):
    """Convert [B, C] logits to probabilities."""
    x = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    name = (normalize or 'softmax').lower()
    if name == 'sigmoid':
        x = np.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x))
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def postprocess_logits(logits, candidates, src_neighb_seq, pop_prior, cfg,
                       src_neighb_times=None, cur_times=None, batch_src=None, cooccur=None):
    """History/pop/cooccur logit boosts then normalize to probs."""
    scores = np.asarray(logits, dtype=np.float64)
    scores = history_boost_logits(
        scores, candidates, src_neighb_seq, src_neighb_times, cur_times, cfg,
        batch_src=batch_src, cooccur=cooccur)
    scores = pop_boost_logits(
        scores, candidates, pop_prior, cfg.get('pop_prior_alpha', 0.0))
    return logits_to_probs(
        scores,
        normalize=cfg.get('pred_normalize', 'softmax'),
        temperature=cfg.get('pred_temperature', 1.0),
    )


def test_val(model, loader, full_neighbor_sampler, num_neighbors, num_neg, min_dst, max_dst, rng,
             csr=None, compute_ap_auc=False):
    """Validate MRR (all negs). Optionally also AP/AUC vs 1st neg."""
    model.eval()
    mrr_eval = MRR_Evaluator()
    ap_list, auc_list, mrr_list = [], [], []
    loader_tqdm = tqdm(loader, ncols=120, desc='Validation')

    for _, batch_data in enumerate(loader_tqdm):
        src_np = np.asarray(batch_data.src.numpy() if hasattr(batch_data.src, 'numpy') else batch_data.src)
        dst_np = np.asarray(batch_data.dst.numpy() if hasattr(batch_data.dst, 'numpy') else batch_data.dst)
        t_np = np.asarray(batch_data.t.numpy() if hasattr(batch_data.t, 'numpy') else batch_data.t)

        if csr is not None:
            src_neighb_seq, src_neighb_interact_times = sample_recent_neighbors_left(
                src_np, t_np, csr, num_neighbors)
        else:
            src_neighb_seq, _, src_neighb_interact_times = full_neighbor_sampler.get_historical_neighbors_left(
                node_ids=src_np, node_interact_times=t_np, num_neighbors=num_neighbors)
        neighbor_num = (src_neighb_seq != 0).sum(axis=1)

        test_dst_np = prepare_batch_dst(
            dst_np, batch_data.neg_dst, num_neg, min_dst, max_dst, rng)

        dst_last_update_time = get_dst_last_update_times(
            full_neighbor_sampler, test_dst_np, t_np, csr=csr)

        pos_score, neg_score = model.predict(
            src_neighb_seq=jt.Var(src_neighb_seq),
            src_neighb_seq_len=jt.Var(neighbor_num),
            src_neighb_interact_times=jt.Var(src_neighb_interact_times),
            cur_pred_times=jt.Var(t_np),
            test_dst=jt.Var(test_dst_np),
            dst_last_update_times=dst_last_update_time)

        pos_np = np.asarray(pos_score).reshape(-1)
        neg_np = np.asarray(neg_score).reshape(len(pos_np), -1)

        if compute_ap_auc:
            y_true = np.concatenate([np.ones_like(pos_np), np.zeros_like(neg_np[:, 0])])
            y_score = np.concatenate([pos_np, neg_np[:, 0]])
            ap_list.append(average_precision_score(y_true, y_score))
            auc_list.append(roc_auc_score(y_true, y_score))

        mrr_list.extend(mrr_eval.eval(pos_np, neg_np))

    out = {'MRR': float(np.mean(mrr_list))}
    if compute_ap_auc and ap_list:
        out['AP'] = float(np.mean(ap_list))
        out['AUC'] = float(np.mean(auc_list))
    return out


def train(model, optimizer, train_loader, val_loader, full_neighbor_sampler, cfg,
          num_epochs, save_path, dataset_name, early_stop_patience, min_dst, max_dst,
          train_dst_np=None, csr=None, init_best_mrr=0.0):
    best_mrr = float(init_best_mrr or 0.0)
    patience_counter = 0
    num_neighbors = cfg['num_neighbors']
    num_neg_train = cfg['num_neg_train']
    num_neg_val = cfg['num_neg_val']
    hard_neg_ratio = cfg.get('hard_neg_ratio', 0.0)
    pop_hard_ratio = cfg.get('pop_hard_ratio', 0.0)
    train_loss_name = cfg.get('train_loss', 'bpr')
    train_softmax_temp = cfg.get('train_softmax_temp', 1.0)
    val_every = max(1, int(cfg.get('val_every', 1) or 1))
    rng_train = np.random.RandomState(42)
    rng_val = np.random.RandomState(123)
    sync_every = int(cfg.get('sync_every', 20) or 20)

    if best_mrr > 0:
        log(f'Resume: init_best_mrr={best_mrr:.6f} (will only save if Val improves)')

    pop_ids, pop_probs = None, None
    if pop_hard_ratio and pop_hard_ratio > 0 and train_dst_np is not None:
        pop_ids, pop_probs = build_popular_dst_table(
            train_dst_np, top_k=cfg.get('pop_top_k', 5000))
        log(f'Popular hard-neg table: {0 if pop_ids is None else len(pop_ids)} nodes')

    feat_kw = dict(
        hard_neg_ratio=hard_neg_ratio,
        pop_ids=pop_ids, pop_probs=pop_probs, pop_hard_ratio=pop_hard_ratio,
        csr=csr,
    )

    log(f'Train loss={train_loss_name} (temp={train_softmax_temp}), '
        f'hard_neg={hard_neg_ratio}, pop_hard={pop_hard_ratio}, '
        f'num_neg_train={num_neg_train}, num_neg_val={num_neg_val} (MRR@100), '
        f'val_every={val_every}, fast_ops={_HAS_NUMBA}, '
        f'JT_SYNC={os.environ.get("JT_SYNC")}, sync_every={sync_every}')

    for epoch in range(num_epochs):
        log(f'\n===== Epoch {epoch + 1}/{num_epochs} start =====')
        model.train()
        train_losses = []
        train_tqdm = tqdm(train_loader, ncols=120, desc=f'Epoch {epoch + 1}')

        # Prefetch: prepare batch N+1 on CPU while GPU runs batch N
        batch_iter = iter(train_tqdm)
        cur_batch = next(batch_iter, None)
        cur_feat = None
        if cur_batch is not None:
            cur_feat = prepare_train_features(
                cur_batch, full_neighbor_sampler, num_neighbors, num_neg_train,
                min_dst, max_dst, rng_train, **feat_kw)

        step = 0
        while cur_batch is not None:
            next_batch = next(batch_iter, None)

            if cur_feat is None:
                cur_batch = next_batch
                if cur_batch is not None:
                    cur_feat = prepare_train_features(
                        cur_batch, full_neighbor_sampler, num_neighbors, num_neg_train,
                        min_dst, max_dst, rng_train, **feat_kw)
                continue

            t_np = cur_feat['t_np']
            test_dst_np = cur_feat['test_dst_np']
            if cur_feat.get('dst_last_np') is not None:
                dst_last_update_time = jt.Var(cur_feat['dst_last_np'])
            else:
                dst_last_update_time = get_dst_last_update_times(
                    full_neighbor_sampler, test_dst_np, t_np, csr=csr)

            pos_score, neg_score = model.predict(
                src_neighb_seq=jt.Var(cur_feat['src_neighb_seq']),
                src_neighb_seq_len=jt.Var(cur_feat['neighbor_num']),
                src_neighb_interact_times=jt.Var(cur_feat['src_neighb_interact_times']),
                cur_pred_times=jt.Var(t_np),
                test_dst=jt.Var(test_dst_np),
                dst_last_update_times=dst_last_update_time)

            bs = test_dst_np.shape[0]
            neg_score = neg_score.reshape(bs, -1)
            loss = compute_train_loss(
                pos_score, neg_score, train_loss_name, train_softmax_temp)

            optimizer.zero_grad()
            optimizer.step(loss)

            # CPU prep next batch while GPU finishes current step (JT_SYNC=0)
            next_feat = None
            if next_batch is not None:
                next_feat = prepare_train_features(
                    next_batch, full_neighbor_sampler, num_neighbors, num_neg_train,
                    min_dst, max_dst, rng_train, **feat_kw)

            step += 1
            if sync_every > 0 and step % sync_every == 0:
                jt.sync_all()
                loss_v = float(loss.item())
                train_losses.append(loss_v)
                train_tqdm.set_description(f'Epoch {epoch + 1}, loss: {loss_v:.4f}')
                # Visible progress even when TQDM is disabled
                if step % max(sync_every, 500) == 0 or step == sync_every:
                    log(f'  step {step}, loss={loss_v:.4f}')
            else:
                train_losses.append(loss)

            cur_batch, cur_feat = next_batch, next_feat

        resolved = []
        for x in train_losses:
            resolved.append(float(x.item()) if hasattr(x, 'item') else float(x))
        train_losses = resolved
        jt.sync_all()

        log(f'Epoch {epoch + 1}, Train Loss: {np.mean(train_losses) if train_losses else float("nan"):.4f}')

        do_val = ((epoch + 1) % val_every == 0) or (epoch + 1 == num_epochs)
        if do_val:
            log(f'Epoch {epoch + 1}, validating...')
            val_res = test_val(
                model, val_loader, full_neighbor_sampler, num_neighbors,
                num_neg_val, min_dst, max_dst, rng_val, csr=csr, compute_ap_auc=False)
            log(f'Epoch {epoch + 1}, Val: {val_res}')

            current_mrr = val_res['MRR']
            if current_mrr > best_mrr:
                best_mrr = current_mrr
                patience_counter = 0
                jt.save(model.state_dict(), f'{save_path}/{dataset_name}_CRAFT_best.pkl')
                log(f'  -> New best MRR: {best_mrr:.6f}, model saved!')
            else:
                patience_counter += 1
                log(f'  -> No improvement for {patience_counter} epoch(s), best MRR: {best_mrr:.6f}')
        else:
            log(f'Epoch {epoch + 1}, skip val (val_every={val_every}), best MRR={best_mrr:.6f}')

        jt.save(model.state_dict(), f'{save_path}/{dataset_name}_CRAFT.pkl')
        log(f'===== Epoch {epoch + 1}/{num_epochs} done (best MRR={best_mrr:.6f}) =====')

        if do_val and patience_counter >= early_stop_patience:
            log(f'\nEarly stopping triggered after {epoch + 1} epochs!')
            log(f'Best validation MRR: {best_mrr:.6f}')
            break

    return best_mrr


def test_competition(model, test_src, test_time, test_candidates, full_neighbor_sampler,
                     num_neighbors, batch_size=200, cfg=None, pop_prior=None,
                     cooccur=None, models=None, model_weights=None, csr=None):
    """Score 100 candidates; optional multi-model logit ensemble before postprocess.

    ``models``: list of models; if set, averages their logits (weighted) then postprocess once.
    """
    cfg = cfg or {}
    if models is None:
        models = [model]
        model_weights = [1.0]
    else:
        if model_weights is None:
            model_weights = [1.0] * len(models)
        wsum = float(sum(model_weights))
        model_weights = [float(w) / wsum for w in model_weights]

    for m in models:
        m.eval()

    all_scores = []
    num_samples = len(test_src)
    num_batches = (num_samples + batch_size - 1) // batch_size

    pbar = tqdm(range(num_batches), ncols=120, desc='Testing')
    for batch_idx in pbar:
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_samples)

        batch_src = test_src[start:end]
        batch_time = test_time[start:end]
        batch_cand = test_candidates[start:end]

        if csr is not None:
            src_neighb_seq, src_neighb_interact_times = sample_recent_neighbors_left(
                batch_src, batch_time, csr, num_neighbors)
        else:
            src_neighb_seq, _, src_neighb_interact_times = full_neighbor_sampler.get_historical_neighbors_left(
                node_ids=batch_src, node_interact_times=batch_time, num_neighbors=num_neighbors)
        neighbor_num = (src_neighb_seq != 0).sum(axis=1)

        test_dst = jt.Var(batch_cand)
        dst_last_update_time = get_dst_last_update_times(
            full_neighbor_sampler, np.asarray(batch_cand), np.asarray(batch_time), csr=csr)

        logits_acc = None
        for m, w in zip(models, model_weights):
            src_neighb_seq_adj = jt.Var(src_neighb_seq) - m.dst_min_idx + 1
            test_dst_adj = test_dst - m.dst_min_idx + 1
            src_neighb_seq_adj = jt.where(
                src_neighb_seq_adj < 0, jt.zeros_like(src_neighb_seq_adj), src_neighb_seq_adj)
            logits = m.forward(
                src_neighb_seq_adj, jt.Var(neighbor_num), jt.Var(src_neighb_interact_times),
                jt.Var(batch_time), test_dst=test_dst_adj, dst_last_update_times=dst_last_update_time)
            logits_np = np.asarray(logits.squeeze(-1)).reshape(len(batch_src), -1).astype(np.float64)
            logits_acc = w * logits_np if logits_acc is None else logits_acc + w * logits_np

        probs = postprocess_logits(
            logits_acc, batch_cand, src_neighb_seq, pop_prior, cfg,
            src_neighb_times=src_neighb_interact_times, cur_times=batch_time,
            batch_src=batch_src, cooccur=cooccur)
        all_scores.append(probs)

    return np.vstack(all_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Model save directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: same as data_dir)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Override config batch size')
    parser.add_argument('--early_stop', type=int, default=10, help='Early stopping patience (on Val MRR)')
    parser.add_argument('--lr', type=float, default=None, help='Override config learning rate')
    parser.add_argument('--num_neg_train', type=int, default=None, help='Override train negatives per positive')
    parser.add_argument('--num_neg_val', type=int, default=None, help='Override val negatives per positive')
    parser.add_argument('--num_neighbors', type=int, default=None, help='Override history neighbor length')
    parser.add_argument('--train_loss', type=str, default=None, choices=['bpr', 'softmax'],
                        help='Override training loss')
    parser.add_argument('--hard_neg_ratio', type=float, default=None,
                        help='Override fraction of hard (history) negatives')
    parser.add_argument('--pop_prior_alpha', type=float, default=None,
                        help='Override logit-space popularity boost at inference')
    parser.add_argument('--history_boost', type=float, default=None,
                        help='Override history candidate logit boost')
    parser.add_argument('--pred_normalize', type=str, default=None, choices=['softmax', 'sigmoid'],
                        help='Override inference normalization')
    parser.add_argument('--hidden_size', type=int, default=None, help='Override hidden size')
    parser.add_argument('--resume', type=str, default=None,
                        help='Load checkpoint before training (path, or "last"/"best")')
    parser.add_argument('--init_best_mrr', type=float, default=0.0,
                        help='Seed best Val MRR when resuming (avoid overwriting better ckpt)')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.data_dir

    cfg = get_dataset_config(args.dataset)
    # CLI overrides
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.lr is not None:
        cfg['lr'] = args.lr
    if args.num_neg_train is not None:
        cfg['num_neg_train'] = args.num_neg_train
    if args.num_neg_val is not None:
        cfg['num_neg_val'] = args.num_neg_val
    if args.num_neighbors is not None:
        cfg['num_neighbors'] = args.num_neighbors
    if args.train_loss is not None:
        cfg['train_loss'] = args.train_loss
    if args.hard_neg_ratio is not None:
        cfg['hard_neg_ratio'] = args.hard_neg_ratio
    if args.pop_prior_alpha is not None:
        cfg['pop_prior_alpha'] = args.pop_prior_alpha
    if args.history_boost is not None:
        cfg['history_boost'] = args.history_boost
    if args.pred_normalize is not None:
        cfg['pred_normalize'] = args.pred_normalize
    if args.hidden_size is not None:
        cfg['hidden_size'] = args.hidden_size

    log('=' * 80)
    log(f'CRAFT Competition (optimized) - Dataset: {args.dataset}')
    log(f'Config: {json.dumps(cfg, indent=2)}')
    log('=' * 80)

    df = pd.read_csv(f'{args.data_dir}/{args.dataset}/train.csv')
    test_df = pd.read_csv(f'{args.data_dir}/{args.dataset}/test.csv')

    src_np = df['src'].values.astype(np.int32)
    dst_np = df['dst'].values.astype(np.int32)
    t_np = df['time'].values.astype(np.int32)
    edge_ids_np = np.arange(len(df), dtype=np.int32) + 1

    test_src = test_df['src'].values.astype(np.int32)
    test_time = test_df['time'].values.astype(np.int32)
    test_candidates = test_df.iloc[:, 2:].values.astype(np.int32)

    log(f'Train+Val: {len(df)}, Test: {len(test_df)}')

    num_total = len(df)
    num_val = int(num_total * 0.15)
    num_train = num_total - num_val

    train_data = TemporalData(
        src=jt.Var(src_np[:num_train]),
        dst=jt.Var(dst_np[:num_train]),
        t=jt.Var(t_np[:num_train]),
        edge_ids=jt.Var(edge_ids_np[:num_train])
    )
    val_data = TemporalData(
        src=jt.Var(src_np[num_train:]),
        dst=jt.Var(dst_np[num_train:]),
        t=jt.Var(t_np[num_train:]),
        edge_ids=jt.Var(edge_ids_np[num_train:])
    )
    full_data = TemporalData(
        src=jt.Var(src_np),
        dst=jt.Var(dst_np),
        t=jt.Var(t_np),
        edge_ids=jt.Var(edge_ids_np)
    )

    train_loader = TemporalDataLoader(
        train_data, batch_size=cfg['batch_size'],
        num_neg_sample=cfg['num_neg_train'], seed=1)
    val_loader = TemporalDataLoader(
        val_data, batch_size=cfg['batch_size'],
        num_neg_sample=cfg['num_neg_val'], seed=2)

    log('Building neighbor sampler (may take a few minutes on large graphs)...')
    full_neighbor_sampler = get_neighbor_sampler(full_data, 'recent', seed=1)
    log('Neighbor sampler ready.')
    log(f'Building CSR fast-ops (numba={_HAS_NUMBA})...')
    csr = build_neighbor_csr(full_neighbor_sampler)
    warmup_fast_ops(csr, num_neighbors=cfg['num_neighbors'])
    log('CSR fast-ops ready.')

    max_node = max(int(src_np.max()), int(dst_np.max()), int(test_candidates.max()))
    node_size = max_node + 1
    dst_min = min(int(dst_np.min()), int(test_candidates.min()))
    dst_max = max(int(dst_np.max()), int(test_candidates.max()))
    src_min = int(src_np.min())

    log(f'Node size: {node_size}, Src min: {src_min}, Dst min/max: {dst_min}/{dst_max}')

    model = build_craft_model(cfg, node_size)
    model.set_min_idx(src_min, dst_min)
    optimizer = jt.nn.Adam(list(model.parameters()), lr=cfg['lr'])

    save_path = args.save_dir
    os.makedirs(save_path, exist_ok=True)

    init_best_mrr = float(args.init_best_mrr or 0.0)
    if args.resume:
        r = args.resume
        if r == 'best':
            r = f'{save_path}/{args.dataset}_CRAFT_best.pkl'
        elif r == 'last':
            r = f'{save_path}/{args.dataset}_CRAFT.pkl'
        if not osp.exists(r):
            raise FileNotFoundError(f'--resume not found: {r}')
        model.load_state_dict(jt.load(r))
        log(f'Resumed weights from {r}')
        if init_best_mrr <= 0 and osp.exists(f'{save_path}/{args.dataset}_CRAFT_best.pkl'):
            # keep best file unless we know MRR; default seed from flag
            pass

    # Persist the exact config used for this run (for predict.py / reproducibility)
    cfg_path = f'{save_path}/{args.dataset}_config.json'
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    log(f'Saved run config to {cfg_path}')

    log(f'\nTraining for {args.epochs} epoch(s), early stop on Val MRR@100 '
        f'(num_neg_val={cfg["num_neg_val"]}, patience={args.early_stop})...')
    best_mrr = train(
        model, optimizer, train_loader, val_loader, full_neighbor_sampler, cfg,
        args.epochs, save_path, args.dataset, args.early_stop, dst_min, dst_max,
        train_dst_np=dst_np[:num_train], csr=csr, init_best_mrr=init_best_mrr)

    log('\nGenerating predictions using best model...')
    best_model_path = f'{save_path}/{args.dataset}_CRAFT_best.pkl'
    if os.path.exists(best_model_path):
        model.load_state_dict(jt.load(best_model_path))
        log(f'Loaded best model from {best_model_path} (best Val MRR={best_mrr:.6f})')
    else:
        model.load_state_dict(jt.load(f'{save_path}/{args.dataset}_CRAFT.pkl'))
        log('Best model not found, using latest model')

    pop_prior = None
    if float(cfg.get('pop_prior_alpha', 0.0) or 0.0) > 0:
        pop_prior = build_dst_log_popularity(dst_np, node_size)
        log(f'Using logit-space pop boost alpha={cfg["pop_prior_alpha"]}')
    cooccur = None
    if float(cfg.get('cooccur_boost', 0.0) or 0.0) > 0:
        log('Building src-dst co-occurrence from train...')
        cooccur = build_src_dst_cooccur(src_np, dst_np)
    log(f'Inference: normalize={cfg.get("pred_normalize")}, '
        f'history_boost={cfg.get("history_boost")}, '
        f'count={cfg.get("history_count_coef")}, '
        f'recency={cfg.get("history_recency_coef")}, '
        f'cooccur={cfg.get("cooccur_boost")}, temp={cfg.get("pred_temperature")}')

    scores = test_competition(
        model, test_src, test_time, test_candidates,
        full_neighbor_sampler, cfg['num_neighbors'], cfg['batch_size'],
        cfg=cfg, pop_prior=pop_prior, cooccur=cooccur, csr=csr)

    log(f'Scores shape: {scores.shape}, range: [{scores.min():.6f}, {scores.max():.6f}]')

    output_file = f'{args.output_dir}/{args.dataset}/{args.dataset}_result.csv'
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for row in scores:
            f.write(','.join([f'{p:.8f}' for p in row]) + '\n')
    log(f'Saved predictions to {output_file}')

    log('\n' + '=' * 80)
    log('DONE')
    log('=' * 80)
