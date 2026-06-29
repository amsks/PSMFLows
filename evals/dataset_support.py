"""evals/dataset_support.py — decode offline-buffer cube xy + KDE.

Pure (numpy/scipy only). The OGBench cube buffer stores raw
`observation[T,28]` / `physics[T,21]` with no labeled cube field, so
the cube-xy slice is identified by matching candidate 2-column windows
against the cube xy already recorded in value_steps from rollouts
(same physical quantity, same workspace). Fail loud if none matches.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _occupancy_iou(cand: np.ndarray, ref: np.ndarray,
                   bins: int = 24) -> float:
    """IoU of *occupied* 2D-histogram cells on the shared union range.

    Binary occupancy (not counts) so a slice is scored on whether it
    covers the same region as `ref`, NOT on matching visitation density
    — rollout cube xy (policy paths) and dataset cube xy (play data)
    share support but have very different densities.
    """
    rx = [min(cand[:, 0].min(), ref[:, 0].min()),
          max(cand[:, 0].max(), ref[:, 0].max())]
    ry = [min(cand[:, 1].min(), ref[:, 1].min()),
          max(cand[:, 1].max(), ref[:, 1].max())]
    hc, _, _ = np.histogram2d(cand[:, 0], cand[:, 1], bins=bins,
                              range=[rx, ry])
    hr, _, _ = np.histogram2d(ref[:, 0], ref[:, 1], bins=bins,
                              range=[rx, ry])
    a, b = hc > 0, hr > 0
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def _match_score(cand: np.ndarray, ref: np.ndarray) -> float:
    """Range-overlap × occupied-cell IoU, in [0, 1]."""
    def overlap(a, b):
        lo, hi = max(a.min(), b.min()), min(a.max(), b.max())
        span = max(a.max(), b.max()) - min(a.min(), b.min())
        return max(0.0, hi - lo) / span if span > 1e-12 else 0.0

    ro = 0.5 * (overlap(cand[:, 0], ref[:, 0])
                + overlap(cand[:, 1], ref[:, 1]))
    return ro * _occupancy_iou(cand, ref)


def _load_buffer_arrays(buffer_dir, n_files: int):
    files = sorted(Path(buffer_dir).glob("episode_*.npz"))[:n_files]
    if not files:
        raise ValueError(f"no episode_*.npz under {buffer_dir}")
    phys, obs = [], []
    for f in files:
        with np.load(f) as z:
            if "physics" in z:
                phys.append(np.asarray(z["physics"], np.float64))
            key = ("observation" if "observation" in z
                   else "observations" if "observations" in z else "obs")
            obs.append(np.asarray(z[key], np.float64))
    return (np.concatenate(phys, 0) if phys else None,
            np.concatenate(obs, 0))


def dataset_cube_xy(buffer_dir, ref_cube_xy: np.ndarray,
                    n_files: int = 200,
                    min_score: float = 0.5) -> np.ndarray:
    """Return decoded offline cube xy `[N,2]`.

    Scans consecutive 2-column windows of `physics` then `observation`,
    scoring each against `ref_cube_xy`; returns the best window if its
    score >= `min_score`, else raises ValueError.
    """
    ref = np.asarray(ref_cube_xy, np.float64).reshape(-1, 2)
    phys, obs = _load_buffer_arrays(buffer_dir, n_files)

    best, best_s = None, -1.0
    for src in (phys, obs):
        if src is None:
            continue
        for j in range(src.shape[1] - 1):
            cand = src[:, j:j + 2]
            if np.ptp(cand[:, 0]) < 1e-9 and np.ptp(cand[:, 1]) < 1e-9:
                continue
            s = _match_score(cand, ref)
            if s > best_s:
                best, best_s = cand, s
    if best is None or best_s < min_score:
        raise ValueError(
            f"no cube-xy slice in buffer matched rollout cube xy "
            f"(best score {best_s:.3f} < {min_score})")
    return np.ascontiguousarray(best)


def support_kde(xy: np.ndarray, grid_x: np.ndarray,
                grid_y: np.ndarray) -> np.ndarray:
    """Gaussian-KDE density on the (grid_y, grid_x) mesh, normalized to
    [0,1]. Falls back to a normalized 2D histogram if KDE is singular.
    Returns `[len(grid_y), len(grid_x)]`."""
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    gx, gy = np.meshgrid(grid_x, grid_y)
    pts = np.vstack([gx.ravel(), gy.ravel()])
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(xy.T)
        z = kde(pts).reshape(gx.shape)
    except Exception:
        h, _, _ = np.histogram2d(
            xy[:, 0], xy[:, 1],
            bins=[len(grid_x), len(grid_y)],
            range=[[grid_x.min(), grid_x.max()],
                   [grid_y.min(), grid_y.max()]])
        z = h.T
    m = z.max()
    return z / m if m > 0 else z
