"""Chunked exact k-nearest-neighbour search over a large point set.

Exact, not approximate: the buffers here are ~1M rows of low-dimensional states, which a
blocked matmul handles in seconds, and an ANN index would add a dependency plus a recall
caveat on every number computed from it.

Memory is bounded by one (M, chunk) distance block, so the point set never has to be
materialised in a squared form -- the pattern the ad-hoc searches in tools/ use, which
builds a full (n, n) Gram and therefore caps n around 4096.
"""

import numpy as np


def neighborhood_sample(points, n_rows, k, seed=0, chunk=100_000, scale=None):
    """Sample rows as whole neighbourhoods rather than independently.

    A uniform sample of a large buffer has neighbourhoods that do not exist: 512 rows drawn
    from 1M leaves each row's "nearest" neighbour far away in state space, so any metric
    about what NEARBY transitions share degenerates into a statement about arbitrary pairs.
    Measured on pointmaze, the number of neighbours covering a latent came out at almost
    exactly k times the single-row rate -- i.e. k independent shots, no neighbourhood
    structure at all.

    This draws `n_rows // k` anchors uniformly and takes each anchor's k nearest rows from
    the WHOLE buffer, so the local density matches what the full dataset would give.
    Neighbourhoods may overlap, so the result can be smaller than `n_rows`.

    Returns:
        Sorted unique row indices.
    """
    n_anchors = max(1, int(n_rows) // int(k))
    rng = np.random.default_rng(seed)
    anchors = rng.choice(len(points), min(n_anchors, len(points)), replace=False)
    idx, _ = knn_indices(points[anchors], points, k, chunk=chunk, scale=scale)
    return np.unique(idx.ravel())


def knn_indices(queries, points, k, chunk=100_000, scale=None, exclude=None):
    """Indices of the `k` nearest `points` to each query, under a per-dimension scaling.

    Args:
        queries: (M, D) query vectors.
        points: (N, D) point set to search; read chunk by chunk.
        k: Neighbours per query (clipped to N).
        chunk: Rows of `points` per block.
        scale: Optional (D,) per-dimension divisor applied to both sides. State dimensions
            carry different units, so an unscaled L2 is dominated by whichever dimension
            happens to have the largest range.
        exclude: Optional (M,) point index per query to drop from its own result (the
            query's own row, when queries are drawn from `points`).

    Returns:
        (idx, dist): (M, k) point indices and their distances, both ascending by distance.
    """
    q = np.asarray(queries, np.float32)
    n = len(points)
    k = min(int(k), n if exclude is None else n - 1)
    if scale is not None:
        scale = np.asarray(scale, np.float32)
        q = q / scale
    if exclude is not None:
        exclude = np.asarray(exclude, np.int64)
        assert exclude.shape == (q.shape[0],), 'exclude must be one point index per query'
    q_sq = (q ** 2).sum(1)[:, None]

    cand_idx, cand_d2 = [], []
    for start in range(0, n, chunk):
        p = np.asarray(points[start:start + chunk], np.float32)
        if scale is not None:
            p = p / scale
        d2 = q_sq + (p ** 2).sum(1)[None] - 2.0 * (q @ p.T)
        if exclude is not None:
            local = exclude - start
            rows = np.nonzero((local >= 0) & (local < p.shape[0]))[0]
            d2[rows, local[rows]] = np.inf
        kk = min(k, p.shape[0])
        part = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        cand_idx.append(part + start)
        cand_d2.append(np.take_along_axis(d2, part, 1))

    idx = np.concatenate(cand_idx, axis=1)
    d2 = np.concatenate(cand_d2, axis=1)
    if d2.shape[1] > k:
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        idx, d2 = np.take_along_axis(idx, part, 1), np.take_along_axis(d2, part, 1)
    order = np.argsort(d2, axis=1)
    idx, d2 = np.take_along_axis(idx, order, 1), np.take_along_axis(d2, order, 1)
    # The expanded form can go slightly negative for coincident points.
    return idx, np.sqrt(np.maximum(d2, 0.0))
