"""Fast CSR + numba kernels for neighbor sampling and last-update times."""
from __future__ import annotations

import numpy as np

try:
    import numba as nb
    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    nb = None
    _HAS_NUMBA = False


def build_neighbor_csr(neighbor_sampler):
    """Pack NeighborSampler per-node arrays into CSR for fast binary search."""
    times_list = neighbor_sampler.nodes_neighbor_times
    ids_list = neighbor_sampler.nodes_neighbor_ids
    n = len(times_list)
    offsets = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        offsets[i + 1] = offsets[i] + int(len(times_list[i]))
    total = int(offsets[-1])
    id_flat = np.empty(total, dtype=np.int64)
    time_flat = np.empty(total, dtype=np.float64)
    for i in range(n):
        a, b = int(offsets[i]), int(offsets[i + 1])
        if a == b:
            continue
        id_flat[a:b] = np.asarray(ids_list[i], dtype=np.int64)
        time_flat[a:b] = np.asarray(times_list[i], dtype=np.float64)
    return offsets, id_flat, time_flat


if _HAS_NUMBA:
    @nb.njit(cache=True)
    def _sample_recent_left_numba(node_ids, times, offsets, id_flat, time_flat,
                                  num_neighbors, out_ids, out_times):
        for i in range(node_ids.shape[0]):
            nid = int(node_ids[i])
            if nid < 0 or nid + 1 >= offsets.shape[0]:
                continue
            t = times[i]
            a = offsets[nid]
            b = offsets[nid + 1]
            lo = a
            hi = b
            while lo < hi:
                mid = (lo + hi) // 2
                if time_flat[mid] < t:
                    lo = mid + 1
                else:
                    hi = mid
            end = lo
            start = end - num_neighbors
            if start < a:
                start = a
            n = end - start
            for j in range(n):
                out_ids[i, j] = id_flat[start + j]
                out_times[i, j] = time_flat[start + j]

    @nb.njit(cache=True)
    def _last_update_numba(node_ids, times, offsets, time_flat, out):
        for i in range(node_ids.shape[0]):
            nid = int(node_ids[i])
            if nid < 0 or nid + 1 >= offsets.shape[0]:
                continue
            t = times[i]
            a = offsets[nid]
            b = offsets[nid + 1]
            if a == b:
                continue
            lo = a
            hi = b
            while lo < hi:
                mid = (lo + hi) // 2
                if time_flat[mid] < t:
                    lo = mid + 1
                else:
                    hi = mid
            if lo > a:
                out[i] = time_flat[lo - 1]


def sample_recent_neighbors_left(node_ids, times, csr, num_neighbors):
    """Recent historical neighbors (left-aligned), matching sampler 'recent' + left."""
    offsets, id_flat, time_flat = csr
    node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    bs = node_ids.shape[0]
    out_ids = np.zeros((bs, num_neighbors), dtype=np.int64)
    out_times = np.zeros((bs, num_neighbors), dtype=np.float32)
    if _HAS_NUMBA:
        _sample_recent_left_numba(
            node_ids, times, offsets, id_flat, time_flat,
            int(num_neighbors), out_ids, out_times)
    else:  # pragma: no cover
        for i in range(bs):
            nid = int(node_ids[i])
            a, b = int(offsets[nid]), int(offsets[nid + 1])
            if a == b:
                continue
            end = int(np.searchsorted(time_flat[a:b], times[i])) + a
            start = max(a, end - num_neighbors)
            n = end - start
            out_ids[i, :n] = id_flat[start:end]
            out_times[i, :n] = time_flat[start:end]
    return out_ids, out_times


def last_update_times(test_dst_np, t_np, csr):
    """Last interaction time before t for each candidate node."""
    offsets, _id_flat, time_flat = csr
    bs, n_cand = test_dst_np.shape
    node_ids = np.asarray(test_dst_np, dtype=np.int64).reshape(-1)
    times = np.broadcast_to(
        np.asarray(t_np, dtype=np.float64).reshape(-1, 1), (bs, n_cand)
    ).reshape(-1)
    out = np.full(node_ids.shape[0], -100000.0, dtype=np.float32)
    if _HAS_NUMBA:
        _last_update_numba(node_ids, times, offsets, time_flat, out)
    else:  # pragma: no cover
        for i in range(node_ids.shape[0]):
            nid = int(node_ids[i])
            a, b = int(offsets[nid]), int(offsets[nid + 1])
            if a == b:
                continue
            j = int(np.searchsorted(time_flat[a:b], times[i]))
            if j > 0:
                out[i] = time_flat[a + j - 1]
    return out.reshape(bs, n_cand)


def warmup_fast_ops(csr, num_neighbors=50):
    """Compile numba kernels once so first train step is not stalling."""
    if not _HAS_NUMBA:
        return
    offsets, id_flat, time_flat = csr
    n = max(1, offsets.shape[0] - 2)
    node_ids = np.array([min(1, n), min(2, n)], dtype=np.int64)
    times = np.array([1.0, 2.0], dtype=np.float64)
    out_ids = np.zeros((2, num_neighbors), dtype=np.int64)
    out_times = np.zeros((2, num_neighbors), dtype=np.float32)
    _sample_recent_left_numba(
        node_ids, times, offsets, id_flat, time_flat,
        int(num_neighbors), out_ids, out_times)
    out = np.zeros(2, dtype=np.float32)
    _last_update_numba(node_ids, times, offsets, time_flat, out)
