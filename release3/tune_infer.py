#!/usr/bin/env python
"""Tune inference-only knobs on Val@100 (1 positive + 99 negatives) without retraining.

Caches model logits once, then grid-searches postprocess configs for best MRR.
"""
import os
import os.path as osp
import sys
import json
import itertools

root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, root)
sys.path.insert(0, osp.join(root, 'JittorGeometric'))
os.environ.setdefault('JT_SYNC', '0')

import argparse
import jittor as jt
import numpy as np
import pandas as pd
from tqdm import tqdm
from jittor_geometric.data import TemporalData
from jittor_geometric.dataloader.temporal_dataloader import get_neighbor_sampler
from jittor_geometric.evaluate.evaluators import MRR_Evaluator

from config import get_dataset_config
from main import (
    build_craft_model, get_dst_last_update_times,
    build_dst_log_popularity, build_src_dst_cooccur, postprocess_logits,
)

jt.flags.use_cuda = 1


def load_cfg(dataset, save_dir):
    defaults = get_dataset_config(dataset)
    path = osp.join(save_dir, f'{dataset}_config.json')
    with open(path) as f:
        cfg = json.load(f)
    # keep model structure from checkpoint; inference knobs will be swept
    return cfg, defaults


def cache_val_logits(models, weights, sampler, src, dst, t, min_dst, max_dst,
                     num_neighbors, batch_size, num_neg, seed=123):
    """Return dict with logits [N,100], candidates, hist, hist_t, cur_t, src, pos col=0."""
    rng = np.random.RandomState(seed)
    mrr_eval = MRR_Evaluator()
    for m in models:
        m.eval()

    n = len(src)
    all_logits, all_cand, all_hist, all_hist_t, all_t, all_src = [], [], [], [], [], []
    num_batches = (n + batch_size - 1) // batch_size
    wsum = float(sum(weights))
    weights = [float(w) / wsum for w in weights]

    for bi in tqdm(range(num_batches), desc='Cache logits', ncols=100):
        s = bi * batch_size
        e = min((bi + 1) * batch_size, n)
        b_src, b_dst, b_t = src[s:e], dst[s:e], t[s:e]
        bs = len(b_src)

        # 1 pos + 99 random negs (collision-free)
        negs = rng.randint(min_dst, max_dst + 1, size=(bs, num_neg)).astype(np.int32)
        for _ in range(8):
            mask = negs == b_dst.reshape(-1, 1)
            if not mask.any():
                break
            negs[mask] = rng.randint(min_dst, max_dst + 1, size=int(mask.sum()))
        cand = np.concatenate([b_dst.reshape(bs, 1), negs], axis=1)

        hist, _, hist_t = sampler.get_historical_neighbors_left(
            node_ids=b_src, node_interact_times=b_t, num_neighbors=num_neighbors)
        neighbor_num = (hist != 0).sum(axis=1)
        dst_lu = get_dst_last_update_times(sampler, cand, b_t)

        logits_acc = None
        for m, w in zip(models, weights):
            hist_adj = jt.Var(hist) - m.dst_min_idx + 1
            hist_adj = jt.where(hist_adj < 0, jt.zeros_like(hist_adj), hist_adj)
            cand_adj = jt.Var(cand) - m.dst_min_idx + 1
            logits = m.forward(
                hist_adj, jt.Var(neighbor_num), jt.Var(hist_t),
                jt.Var(b_t), test_dst=cand_adj, dst_last_update_times=dst_lu)
            arr = np.asarray(logits.squeeze(-1)).reshape(bs, -1).astype(np.float64)
            logits_acc = w * arr if logits_acc is None else logits_acc + w * arr

        all_logits.append(logits_acc)
        all_cand.append(cand)
        all_hist.append(hist)
        all_hist_t.append(hist_t)
        all_t.append(b_t)
        all_src.append(b_src)

    return {
        'logits': np.vstack(all_logits),
        'cand': np.vstack(all_cand),
        'hist': np.vstack(all_hist),
        'hist_t': np.vstack(all_hist_t),
        't': np.concatenate(all_t),
        'src': np.concatenate(all_src),
    }


def eval_mrr(cache, pop_prior, cooccur, infer_cfg, max_rows=None):
    mrr_eval = MRR_Evaluator()
    logits = cache['logits']
    n = len(logits) if max_rows is None else min(max_rows, len(logits))
    probs = postprocess_logits(
        logits[:n], cache['cand'][:n], cache['hist'][:n], pop_prior, infer_cfg,
        src_neighb_times=cache['hist_t'][:n], cur_times=cache['t'][:n],
        batch_src=cache['src'][:n], cooccur=cooccur)
    # pos is column 0
    pos = probs[:, 0]
    neg = probs[:, 1:]
    return float(np.mean(mrr_eval.eval(pos, neg)))


def sweep(dataset, data_dir, save_dir, max_val=8000, batch_size=100):
    cfg, defaults = load_cfg(dataset, save_dir)
    df = pd.read_csv(f'{data_dir}/{dataset}/train.csv')
    src_np = df['src'].values.astype(np.int32)
    dst_np = df['dst'].values.astype(np.int32)
    t_np = df['time'].values.astype(np.int32)

    num_total = len(df)
    num_val = int(num_total * 0.15)
    num_train = num_total - num_val
    # use a subset of val for speed
    v_src = src_np[num_train:num_train + max_val]
    v_dst = dst_np[num_train:num_train + max_val]
    v_t = t_np[num_train:num_train + max_val]

    full = TemporalData(
        src=jt.Var(src_np), dst=jt.Var(dst_np), t=jt.Var(t_np),
        edge_ids=jt.Var(np.arange(len(df), dtype=np.int32) + 1))
    sampler = get_neighbor_sampler(full, 'recent', seed=1)

    max_node = max(int(src_np.max()), int(dst_np.max())) + 1
    node_size = max_node
    dst_min, dst_max = int(dst_np.min()), int(dst_np.max())
    src_min = int(src_np.min())

    best_path = osp.join(save_dir, f'{dataset}_CRAFT_best.pkl')
    last_path = osp.join(save_dir, f'{dataset}_CRAFT.pkl')
    models, weights = [], []
    for p, wkey, default_w in (
        (best_path, 'ensemble_best_weight', 0.65),
        (last_path, 'ensemble_last_weight', 0.35),
    ):
        if not osp.exists(p):
            continue
        m = build_craft_model(cfg, node_size)
        m.set_min_idx(src_min, dst_min)
        m.load_state_dict(jt.load(p))
        models.append(m)
        weights.append(float(defaults.get(wkey, default_w)))
    if len(models) == 1:
        weights = [1.0]
    print(f'{dataset}: {len(models)} ckpt(s), val_rows={len(v_src)}')

    cache = cache_val_logits(
        models, weights, sampler, v_src, v_dst, v_t, dst_min, dst_max,
        cfg['num_neighbors'], batch_size, num_neg=99)

    pop_prior = build_dst_log_popularity(dst_np, node_size)
    print('Building cooccur...')
    cooccur = build_src_dst_cooccur(src_np, dst_np)

    # grids — dataset-specific emphasis
    if dataset == 'dataset1':
        grid = {
            'history_boost': [0.0, 2.0, 3.5, 5.0],
            'history_count_coef': [0.0, 0.8, 1.5],
            'history_recency_coef': [0.0, 1.5, 3.0],
            'cooccur_boost': [0.0, 1.0, 2.0],
            'pop_prior_alpha': [0.0, 0.3, 0.8],
            'pred_temperature': [0.7, 0.85, 1.0],
        }
    else:
        # ds2: new-link heavy — try milder history, stronger pop / cooler temp
        grid = {
            'history_boost': [0.0, 0.5, 1.5, 2.5],
            'history_count_coef': [0.0, 0.5, 1.0],
            'history_recency_coef': [0.0, 1.0, 2.0],
            'cooccur_boost': [0.0, 0.5, 1.2],
            'pop_prior_alpha': [0.0, 0.8, 1.5, 2.5],
            'pred_temperature': [0.7, 0.9, 1.1],
        }

    keys = list(grid.keys())
    best = None
    best_mrr = -1.0
    # baseline: raw softmax no boost
    base_cfg = {**{k: 0.0 for k in keys}, 'pred_normalize': 'softmax', 'pred_temperature': 1.0}
    base_mrr = eval_mrr(cache, pop_prior, cooccur, base_cfg)
    print(f'baseline (no boost, T=1): MRR@100={base_mrr:.6f}')

    # random sample of combinations to keep runtime sane
    combos = list(itertools.product(*[grid[k] for k in keys]))
    rng = np.random.RandomState(0)
    if len(combos) > 80:
        idx = rng.choice(len(combos), size=80, replace=False)
        combos = [combos[i] for i in idx]
    # always include current production defaults
    prod = tuple(defaults.get(k, 0.0) for k in keys)
    if prod not in combos:
        combos.append(prod)

    for vals in tqdm(combos, desc='Sweep', ncols=100):
        infer = {'pred_normalize': 'softmax'}
        for k, v in zip(keys, vals):
            infer[k] = v
        mrr = eval_mrr(cache, pop_prior, cooccur, infer)
        if mrr > best_mrr:
            best_mrr = mrr
            best = dict(infer)

    print(f'BEST MRR@100={best_mrr:.6f}  delta={best_mrr - base_mrr:+.6f}')
    print(json.dumps(best, indent=2))

    out_path = osp.join(save_dir, f'{dataset}_infer_best.json')
    with open(out_path, 'w') as f:
        json.dump({'mrr_at_100': best_mrr, 'baseline_mrr': base_mrr, 'infer': best}, f, indent=2)
    print(f'Saved {out_path}')
    return best, best_mrr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--max_val', type=int, default=6000)
    parser.add_argument('--batch_size', type=int, default=100)
    args = parser.parse_args()
    sweep(args.dataset, args.data_dir, args.save_dir, args.max_val, args.batch_size)


if __name__ == '__main__':
    main()
