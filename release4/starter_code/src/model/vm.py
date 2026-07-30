import os
from math import ceil
from typing import Dict, List, Optional

import jittor as jt
import numpy as np

from .feature import FeatureExtraction, Decoder
from .spec import ModelSpec

from ..data.asset import Asset


def get_random_indices(n, m):
    m = min(int(m), int(n))
    assert m > 0
    if m == n:
        return jt.array(np.arange(n)).int32()
    idx = np.random.permutation(n)[:m]
    return jt.array(idx).int32()


def pairwise_sqdist(a, b):
    """a: (B,N,3), b: (B,M,3) → (B,N,M)"""
    return ((a.unsqueeze(2) - b.unsqueeze(1)) ** 2).sum(-1)


def normalize_to_unit_sphere_jt(pc, ref):
    """
    Normalize pc with center/scale from ref — matches evaluate.normalize_to_unit_sphere.
    pc, ref: (B, N, 3) / (B, M, 3)
    """
    rmax = ref.max(dim=1, keepdims=True)
    rmin = ref.min(dim=1, keepdims=True)
    center = (rmax + rmin) * 0.5
    centered = ref - center
    rad = (centered ** 2).sum(dim=-1).sqrt()  # (B, M)
    scale = rad.max(dim=1, keepdims=True).unsqueeze(-1)  # (B,1,1)
    scale = jt.maximum(scale, jt.array(1e-12))
    return (pc - center) / scale


def chamfer_sq(a, b):
    """Symmetric Chamfer (mean of squared NN distances), differentiable."""
    d = pairwise_sqdist(a, b)
    a2b, _ = jt.topk(d, k=1, dim=-1, largest=False)
    b2a, _ = jt.topk(d.transpose(0, 2, 1), k=1, dim=-1, largest=False)
    return a2b.mean() + b2a.mean()


def onesided_chamfer_sq(a, b):
    """One-sided Chamfer a→b (P2S proxy when b is surface samples)."""
    d = pairwise_sqdist(a, b)
    a2b, _ = jt.topk(d, k=1, dim=-1, largest=False)
    return a2b.mean()


def repulsion_hinge(pc, k: int = 8, radius: float = 0.03):
    """
    Penalize neighbors closer than `radius` (in same coordinate frame as pc).
    Encourages local spacing → protects bidirectional CD from clustering.
    """
    B, N, _ = pc.shape
    kk = min(int(k) + 1, N)
    d = pairwise_sqdist(pc, pc)  # (B,N,N)
    knn_d, _ = jt.topk(d, k=kk, dim=-1, largest=False)
    knn_d = knn_d[:, :, 1:]  # drop self
    r2 = float(radius) * float(radius)
    return jt.nn.relu(r2 - knn_d).mean()


class VelocityModule(ModelSpec):

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)

        cfg = self.model_config
        # geometry
        self.frame_knn = cfg['frame_knn']
        self.num_train_points = cfg['num_train_points']
        # CD/P2S/repulse may use more points than MSE (default: all selected)
        self.num_cd_points = int(cfg.get('num_cd_points', self.num_train_points))
        self.cd_unit_sphere = bool(cfg.get('cd_unit_sphere', True))
        self.repulsion_k = int(cfg.get('repulsion_k', 8))
        self.repulsion_radius = float(cfg.get('repulsion_radius', 0.03))

        # score-matching
        self.dsm_sigma = cfg['dsm_sigma']

        # training target: full displacement (clean-noisy) or residual (clean-mix)
        # residual + single-step infer is the intended train/infer contract
        self.target_mode = cfg.get('target_mode', 'full')

        # inference knobs (conservative defaults: avoid CD collapse from over-denoise)
        pred = cfg.get('predict', {}) or {}
        self.pred_outer_steps = int(pred.get('outer_steps', 1))
        self.pred_inner_steps = int(pred.get('inner_steps', 1))
        self.pred_patch_size = int(pred.get('patch_size', 1000))
        self.pred_seed_k = float(pred.get('seed_k', 8))
        self.pred_seed_k_alpha = float(pred.get('seed_k_alpha', 0.5))
        self.pred_fusion = str(pred.get('fusion', 'weighted'))  # weighted | argmax
        self.pred_step_scale = float(pred.get('step_scale', 0.6))
        self.pred_seed_mode = str(pred.get('seed_mode', 'fps_fast'))  # fps | fps_fast | random
        self.pred_fps_candidates = int(pred.get('fps_candidates', 8192))
        # residual multi-step: split correction across inner steps (avoids overshoot)
        self.pred_residual_split = bool(pred.get('residual_split', False))
        # hard cap GPU patch batch (avoids OOM on large seed_k)
        self.pred_max_patch_batch = int(pred.get('max_patch_batch', 24))
        # blend with input: out = noisy + blend_alpha * (pred - noisy)
        # <1 recovers CD distribution; 1.0 = full network output
        self.pred_blend_alpha = float(pred.get('blend_alpha', 0.65))
        # optional multi-ckpt / TTA averaging at predict time
        ens = pred.get('ensemble_ckpts', None) or []
        self.pred_ensemble_ckpts = [str(p) for p in ens if p]
        # env overrides (used by run_all.sh / sweep)
        ens_env = os.environ.get('ENSEMBLE_CKPTS', '').strip()
        if ens_env:
            self.pred_ensemble_ckpts = [p.strip() for p in ens_env.split(',') if p.strip()]
        self.pred_tta_runs = max(1, int(pred.get('tta_runs', 1)))
        tta_env = os.environ.get('TTA_RUNS', '').strip()
        if tta_env:
            self.pred_tta_runs = max(1, int(tta_env))
        for env_key, attr, cast in (
            ('PREDICT_BLEND_ALPHA', 'pred_blend_alpha', float),
            ('PREDICT_STEP_SCALE', 'pred_step_scale', float),
            ('PREDICT_OUTER_STEPS', 'pred_outer_steps', int),
            ('PREDICT_INNER_STEPS', 'pred_inner_steps', int),
            ('PREDICT_SEED_K', 'pred_seed_k', float),
        ):
            v = os.environ.get(env_key, '').strip()
            if v:
                setattr(self, attr, cast(v))

        # networks
        self.encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=cfg['feat_embedding_dim']
        )

        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=3,
            hidden_size=cfg['decoder_hidden_dim'],
        )

    def get_supervised_loss(self, pc_noisy, pc_mix, pc_clean):
        """
        Returns dict:
          loss:    displacement MSE
          cd:      symmetric Chamfer (unit-sphere, eval-aligned) after prediction
          p2s:     one-sided Chamfer pred→clean (surface proxy)
          repulse: local spacing hinge (anti-clustering)
        """
        B, N_noisy, d = pc_mix.shape

        n_mse = min(int(self.num_train_points), N_noisy)
        n_cd = min(max(int(self.num_cd_points), n_mse), N_noisy)
        # one index set: first n_mse for MSE, first n_cd for geometry (same pool)
        pnt_idx = get_random_indices(N_noisy, n_cd)

        feat = self.encoder(pc_mix)  # (B, N, F)
        F_dim = feat.shape[2]

        feat_s = feat[:, pnt_idx, :]
        pc_noisy_s = pc_noisy[:, pnt_idx, :]
        pc_mix_s = pc_mix[:, pnt_idx, :]
        pc_clean_s = pc_clean[:, pnt_idx, :]

        # MSE on a subset if n_cd > n_mse (keeps DSM cost moderate)
        mse_idx = slice(0, n_mse)
        feat_m = feat_s[:, mse_idx, :]
        pc_noisy_m = pc_noisy_s[:, mse_idx, :]
        pc_mix_m = pc_mix_s[:, mse_idx, :]
        pc_clean_m = pc_clean_s[:, mse_idx, :]

        if self.target_mode == 'residual':
            target_m = pc_clean_m - pc_mix_m
        else:
            target_m = pc_clean_m - pc_noisy_m

        pred_dir_m = self.decoder(
            c=feat_m.reshape(-1, F_dim)
        ).reshape(B, n_mse, d)

        loss = (((pred_dir_m - target_m) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()

        # full selected set for geometric losses (single forward if n_cd == n_mse)
        if n_cd == n_mse:
            pred_dir_s = pred_dir_m
        else:
            pred_dir_s = self.decoder(
                c=feat_s.reshape(-1, F_dim)
            ).reshape(B, n_cd, d)

        if self.target_mode == 'residual':
            pc_pred = pc_mix_s + pred_dir_s
        else:
            pc_pred = pc_noisy_s + pred_dir_s

        if self.cd_unit_sphere:
            ref = pc_clean_s
            pc_pred_n = normalize_to_unit_sphere_jt(pc_pred, ref)
            pc_clean_n = normalize_to_unit_sphere_jt(pc_clean_s, ref)
        else:
            pc_pred_n = pc_pred
            pc_clean_n = pc_clean_s

        cd = chamfer_sq(pc_pred_n, pc_clean_n)
        p2s = onesided_chamfer_sq(pc_pred_n, pc_clean_n)
        repulse = repulsion_hinge(
            pc_pred_n, k=self.repulsion_k, radius=self.repulsion_radius
        )

        return {"loss": loss, "cd": cd, "p2s": p2s, "repulse": repulse}

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: Optional[int] = None, step_scale: Optional[float] = None):
        """
        pcl_noisy: (B, N, 3)
        """
        if num_steps is None:
            num_steps = self.pred_inner_steps
        if step_scale is None:
            step_scale = self.pred_step_scale

        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            for _ in range(num_steps):
                feat = self.encoder(pcl_next)  # (B, N, F)
                F_dim = feat.shape[2]

                pred_dir = self.decoder(
                    c=feat.reshape(-1, F_dim)
                ).reshape(B, N, d)

                if self.target_mode == 'residual':
                    if self.pred_residual_split and num_steps > 1:
                        pcl_next = pcl_next + (step_scale / float(num_steps)) * pred_dir
                    else:
                        pcl_next = pcl_next + step_scale * pred_dir
                else:
                    pcl_next = pcl_next + (step_scale / float(num_steps)) * pred_dir
        return pcl_next, None

    def _denoise_cloud(self, pc_noisy) -> np.ndarray:
        """Run outer patch-based denoise loops on a single (N,3) cloud."""
        pc_next = pc_noisy
        for _ in range(self.pred_outer_steps):
            pc_next = patch_based_denoise(
                model=self,
                pcl_noisy=pc_next,
                patch_size=self.pred_patch_size,
                seed_k=self.pred_seed_k,
                seed_k_alpha=self.pred_seed_k_alpha,
                fusion=self.pred_fusion,
                inner_steps=self.pred_inner_steps,
                step_scale=self.pred_step_scale,
                seed_mode=self.pred_seed_mode,
                fps_candidates=self.pred_fps_candidates,
                max_patch_batch=self.pred_max_patch_batch,
            )
            if pc_next is None:
                raise RuntimeError('patch_based_denoise failed')
        out = pc_next.detach().numpy() if isinstance(pc_next, jt.Var) else np.asarray(pc_next)
        return out.astype(np.float32, copy=False)

    def _blend_with_noisy(self, pc_noisy, pc_pred: np.ndarray) -> np.ndarray:
        """out = noisy + α (pred - noisy). α=1 keeps full denoise; α<1 protects CD."""
        a = float(self.pred_blend_alpha)
        if a >= 1.0 - 1e-8:
            return pc_pred.astype(np.float32, copy=False)
        noisy = pc_noisy.numpy() if isinstance(pc_noisy, jt.Var) else np.asarray(pc_noisy)
        noisy = noisy.astype(np.float64, copy=False)
        pred = pc_pred.astype(np.float64, copy=False)
        a = max(0.0, min(1.0, a))
        return (noisy + a * (pred - noisy)).astype(np.float32)

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        pc_mix = batch['pc_mix'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)
        return self.get_supervised_loss(
            pc_noisy=pc_noisy,
            pc_mix=pc_mix,
            pc_clean=pc_clean,
        )

    def execute(self, **kwargs) -> Dict:  # type: ignore
        return self.training_step(**kwargs)

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        assert pc_noisy_batch.ndim == 3

        # Ensemble: average across configured checkpoints (+ optional TTA).
        # Empty ensemble → use currently loaded weights only.
        if self.pred_ensemble_ckpts:
            ckpt_list = list(self.pred_ensemble_ckpts)
        else:
            ckpt_list = [None]

        seen = set()
        uniq = []
        for p in ckpt_list:
            key = os.path.abspath(p) if p is not None else '__current__'
            if key in seen:
                continue
            seen.add(key)
            uniq.append(p)
        ckpt_list = uniq

        res = []
        for pc_noisy in pc_noisy_batch:
            acc = None
            n_avg = 0
            for ckpt in ckpt_list:
                if ckpt is not None:
                    self.load(ckpt)
                for _ in range(self.pred_tta_runs):
                    out = self._denoise_cloud(pc_noisy)
                    if acc is None:
                        acc = out.astype(np.float64, copy=True)
                    else:
                        acc += out.astype(np.float64, copy=False)
                    n_avg += 1
            assert acc is not None and n_avg > 0
            pred = (acc / float(n_avg)).astype(np.float32)
            pred = self._blend_with_noisy(pc_noisy, pred)
            res.append({"pc_denoised": pred})
        return res

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                res.append({
                    "pc_noisy": b.meta['pc_noisy'],
                    "pc_clean": b.meta['pc_clean'],
                    "pc_mix": b.meta['pc_mix'],
                })
            else:
                d = {
                    "pc_noisy": b.sampled_vertices_noisy,
                }
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res


def farthest_point_sampling_np(pts: np.ndarray, num_pnts: int) -> np.ndarray:
    """FPS on (N,3) float array → indices (num_pnts,)."""
    N = pts.shape[0]
    num_pnts = max(1, min(int(num_pnts), N))
    selected = np.empty(num_pnts, dtype=np.int64)
    dist = np.full(N, 1e10, dtype=np.float64)
    farthest = 0
    pts64 = pts.astype(np.float64, copy=False)
    for i in range(num_pnts):
        selected[i] = farthest
        centroid = pts64[farthest]
        d = np.sum((pts64 - centroid) ** 2, axis=1)
        np.minimum(dist, d, out=dist)
        farthest = int(np.argmax(dist))
    return selected


def farthest_point_sampling(pcls, num_pnts):
    """Fast numpy FPS. pcls: (B,N,3) → sampled (B,K,3), indices (B,K)."""
    B, N, _ = pcls.shape
    pts_all = pcls.numpy() if isinstance(pcls, jt.Var) else np.asarray(pcls)
    sampled_list = []
    indices_list = []
    for b in range(B):
        idx = farthest_point_sampling_np(pts_all[b], num_pnts)
        sampled_list.append(pts_all[b][idx][None, ...])
        indices_list.append(idx[None, ...])
    sampled = jt.array(np.concatenate(sampled_list, axis=0).astype(np.float32))
    indices = jt.array(np.concatenate(indices_list, axis=0).astype(np.int32))
    return sampled, indices


def fps_fast_sampling(pcls, num_pnts, max_candidates: int = 8192):
    """
    Approximate FPS: random-downsample to max_candidates, then FPS.
    Much faster on large clouds (50k → 8k) with similar coverage.
    """
    B, N, _ = pcls.shape
    pts_all = pcls.numpy() if isinstance(pcls, jt.Var) else np.asarray(pcls)
    num_pnts = max(1, min(int(num_pnts), N))
    sampled_list = []
    indices_list = []
    for b in range(B):
        pts = pts_all[b]
        C = min(N, max(num_pnts, int(max_candidates)))
        if C < N:
            cand = np.random.choice(N, size=C, replace=False)
            local = farthest_point_sampling_np(pts[cand], num_pnts)
            idx = cand[local]
        else:
            idx = farthest_point_sampling_np(pts, num_pnts)
        sampled_list.append(pts[idx][None, ...])
        indices_list.append(idx[None, ...])
    sampled = jt.array(np.concatenate(sampled_list, axis=0).astype(np.float32))
    indices = jt.array(np.concatenate(indices_list, axis=0).astype(np.int32))
    return sampled, indices


def random_point_sampling(pcls, num_pnts):
    """Fast random seed sampling (approximate coverage)."""
    B, N, _ = pcls.shape
    num_pnts = max(1, min(int(num_pnts), N))
    pts_all = pcls.numpy() if isinstance(pcls, jt.Var) else np.asarray(pcls)
    sampled_list = []
    indices_list = []
    for b in range(B):
        idx = np.random.choice(N, size=num_pnts, replace=False).astype(np.int64)
        sampled_list.append(pts_all[b][idx][None, ...])
        indices_list.append(idx[None, ...])
    sampled = jt.array(np.concatenate(sampled_list, axis=0).astype(np.float32))
    indices = jt.array(np.concatenate(indices_list, axis=0).astype(np.int32))
    return sampled, indices


def knn_points(x, y, k):
    """
    x: (B, P, 3)
    y: (B, N, 3)
    """
    N = y.shape[1]
    k = min(int(k), N)
    dist = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
    dist_k, idx = jt.topk(dist, k=k, dim=-1, largest=False)
    B = x.shape[0]
    nn = []
    for b in range(B):
        nn.append(y[b][idx[b]])
    nn = jt.stack(nn, dim=0)
    return dist_k, idx, nn


def knn_patches_ckdtree(pc: np.ndarray, seeds: np.ndarray, k: int):
    """
    CPU cKDTree patch build.
    pc: (N,3), seeds: (P,3) → dists (P,k), idxs (P,k), patches (P,k,3)
    """
    from scipy.spatial import cKDTree
    k = min(int(k), pc.shape[0])
    tree = cKDTree(pc)
    dists, idxs = tree.query(seeds, k=k)
    if k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]
    patches = pc[idxs]
    return dists.astype(np.float32), idxs.astype(np.int64), patches.astype(np.float32)


def patch_based_denoise(
    model: VelocityModule,
    pcl_noisy,
    patch_size=1000,
    seed_k=6,
    seed_k_alpha=1,
    fusion='weighted',
    inner_steps=4,
    step_scale=1.0,
    seed_mode='fps_fast',
    fps_candidates=8192,
    max_patch_batch=32,
) -> Optional[jt.Var]:
    """
    pcl_noisy: (N, 3)
    fusion: 'weighted' | 'argmax'
    seed_mode: 'fps' | 'fps_fast' | 'random'
    """
    assert len(pcl_noisy.shape) == 2

    # Stay on numpy for seed/patch construction (avoids GPU idle + huge pairwise mats)
    pc_np = pcl_noisy.numpy() if isinstance(pcl_noisy, jt.Var) else np.asarray(pcl_noisy, dtype=np.float32)
    if pc_np.dtype != np.float32:
        pc_np = pc_np.astype(np.float32)
    N = pc_np.shape[0]
    patch_size = min(int(patch_size), N)
    num_patches = max(1, int(seed_k * N / patch_size))
    num_patches = min(num_patches, N)

    if seed_mode == 'random':
        seed_idx = np.random.choice(N, size=num_patches, replace=False).astype(np.int64)
    elif seed_mode == 'fps':
        seed_idx = farthest_point_sampling_np(pc_np, num_patches)
    else:  # fps_fast (default)
        C = min(N, max(num_patches, int(fps_candidates)))
        if C < N:
            cand = np.random.choice(N, size=C, replace=False)
            local = farthest_point_sampling_np(pc_np[cand], num_patches)
            seed_idx = cand[local]
        else:
            seed_idx = farthest_point_sampling_np(pc_np, num_patches)

    seeds = pc_np[seed_idx]  # (P, 3)
    num_patches = seeds.shape[0]
    patch_dists, point_idxs, patches = knn_patches_ckdtree(pc_np, seeds, patch_size)
    # patch_dists: Euclidean; normalize within patch by farthest neighbor
    denom = patch_dists[:, -1:] + 1e-8
    patch_dists_norm = patch_dists / denom

    # center patches by seed
    patches_centered = patches - seeds[:, None, :]

    # denoise in GPU batches (hard-capped to avoid OOM on Dynamic EdgeConv kNN)
    patch_step = max(1, int(ceil(N / (max(seed_k_alpha, 1e-6) * patch_size))))
    patch_step = min(num_patches, max(1, min(patch_step, int(max_patch_batch))))

    patches_denoised = np.empty_like(patches_centered)
    i = 0
    while i < num_patches:
        curr_np = patches_centered[i:i + patch_step]
        curr = jt.array(curr_np)
        try:
            out, _ = model.denoise_langevin_dynamics(
                curr, num_steps=inner_steps, step_scale=step_scale
            )
        except Exception as e:
            print("Denoise error:", e)
            return None
        patches_denoised[i:i + patch_step] = out.numpy()
        i += patch_step

    # restore world coords
    patches_denoised = patches_denoised + seeds[:, None, :]

    if fusion == 'argmax':
        # best patch per point by normalized distance
        best_dist = np.full(N, 1e10, dtype=np.float32)
        best_xyz = np.zeros((N, 3), dtype=np.float32)
        for p in range(num_patches):
            g = point_idxs[p]
            d = patch_dists_norm[p]
            better = d < best_dist[g]
            if np.any(better):
                gg = g[better]
                best_dist[gg] = d[better]
                best_xyz[gg] = patches_denoised[p][better]
        uncovered = best_dist >= 1e9
        if np.any(uncovered):
            best_xyz[uncovered] = pc_np[uncovered]
        return jt.array(best_xyz)

    # weighted soft fusion
    pcl_sum = np.zeros((N, 3), dtype=np.float64)
    w_sum = np.zeros((N,), dtype=np.float64)
    for p in range(num_patches):
        w = np.exp(-patch_dists_norm[p].astype(np.float64))
        g = point_idxs[p]
        pcl_sum[g] += patches_denoised[p].astype(np.float64) * w[:, None]
        w_sum[g] += w

    valid = w_sum > 0
    pcl_out = np.zeros((N, 3), dtype=np.float32)
    pcl_out[valid] = (pcl_sum[valid] / w_sum[valid, None]).astype(np.float32)
    if not np.all(valid):
        pcl_out[~valid] = pc_np[~valid]
    return jt.array(pcl_out)
