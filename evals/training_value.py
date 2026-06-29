"""evals/training_value.py — pure helpers for the training-data value
analysis (numpy/pandas only; no torch/jax)."""

from __future__ import annotations

import numpy as np

from evals._profile_core import _spearman


def region_labels(grip: np.ndarray, lift: np.ndarray, thr) -> np.ndarray:
    """Per-state phase: 'transport' if lifted+gripped, else 'grasp' if
    gripped, else 'reach'. Mirrors evals._profile_core._region (lift->grasp)."""
    grip = np.asarray(grip, np.float64).reshape(-1)
    lift = np.asarray(lift, np.float64).reshape(-1)
    transport = (lift > thr.delta_lift) & (grip > thr.tau_grip)
    return np.where(transport, "transport",
                    np.where(grip > thr.tau_grip, "grasp", "reach"))


def cube_to_goal_dist(cube_xyz: np.ndarray, goal_xyz: np.ndarray) -> np.ndarray:
    """Euclidean cube-to-goal distance per row."""
    cube = np.asarray(cube_xyz, np.float64).reshape(-1, 3)
    goal = np.asarray(goal_xyz, np.float64).reshape(3)
    return np.linalg.norm(cube - goal, axis=1)


def phase_spearman_table(df, value_col: str = "V", group_cols=("task",)):
    """Per (region) Spearman rho(value, -d), aggregated mean/std over the
    `group_cols` combinations (default per-task; pass ('pair','task') for a
    multi-seed run). `df` has columns region, d, <value_col>, + group_cols."""
    import pandas as pd

    group_cols = list(group_cols)
    rows = []
    for region, g in df.groupby("region"):
        vals = []
        for _, gg in g.groupby(group_cols):
            vals.append(_spearman(gg[value_col].to_numpy(),
                                  -gg["d"].to_numpy()))
        vals = np.array(vals, dtype=float)
        rows.append({"region": region,
                     "rho_mean": float(np.nanmean(vals)),
                     "rho_std": float(np.nanstd(vals)),
                     "n_groups": int(np.isfinite(vals).sum())})
    return pd.DataFrame(rows)


def horizon_reach_label(d_future: np.ndarray, thresh: float,
                        horizon: int) -> bool:
    """True iff the cube comes within `thresh` of the goal within the first
    `horizon` future steps. `d_future` = cube->goal distances from step t
    onward (within one episode). Empty tail -> False."""
    seg = np.asarray(d_future, np.float64).reshape(-1)[:horizon]
    return bool(seg.size and np.min(seg) < thresh)


def flow_step_labels(cube_traj: np.ndarray, goals_xyz: np.ndarray,
                     horizon: int, thresh: float) -> np.ndarray:
    """[T, n_goals] bool success-bound per step per goal for ONE episode:
    does the cube reach within `thresh` of each goal within `horizon` future
    steps. `cube_traj` = [T,3] cube xyz; `goals_xyz` = [n_goals,3]."""
    cube_traj = np.asarray(cube_traj, np.float64).reshape(-1, 3)
    goals_xyz = np.asarray(goals_xyz, np.float64).reshape(-1, 3)
    T, G = len(cube_traj), len(goals_xyz)
    out = np.zeros((T, G), dtype=bool)
    for gi in range(G):
        d = np.linalg.norm(cube_traj - goals_xyz[gi], axis=1)  # [T]
        for t in range(T):
            out[t, gi] = horizon_reach_label(d[t:], thresh, horizon)
    return out


def outcome_spearman_table(df, value_col: str = "V", group_cols=("task",)):
    """Per (region, outcome) Spearman rho(value, -d), mean/std over the
    `group_cols` combinations. `df` has columns region, outcome, d,
    <value_col>, + group_cols."""
    import pandas as pd

    group_cols = list(group_cols)
    rows = []
    for (region, outcome), g in df.groupby(["region", "outcome"]):
        vals = []
        for _, gg in g.groupby(group_cols):
            vals.append(_spearman(gg[value_col].to_numpy(),
                                  -gg["d"].to_numpy()))
        vals = np.array(vals, dtype=float)
        rows.append({"region": region, "outcome": outcome,
                     "rho_mean": float(np.nanmean(vals))
                     if np.isfinite(vals).any() else float("nan"),
                     "rho_std": float(np.nanstd(vals))
                     if np.isfinite(vals).any() else float("nan"),
                     "n_groups": int(np.isfinite(vals).sum())})
    return pd.DataFrame(rows)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), tie-aware; vectorized O(n log n)."""
    a = np.asarray(a, np.float64)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    is_new = np.empty(len(a), bool)
    is_new[0] = True
    is_new[1:] = sa[1:] != sa[:-1]
    grp = np.cumsum(is_new) - 1
    pos = np.arange(1, len(a) + 1, dtype=np.float64)
    mean_r = np.bincount(grp, weights=pos) / np.bincount(grp)
    ranks = np.empty(len(a), np.float64)
    ranks[order] = mean_r[grp]
    return ranks


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC = P(value(pos) > value(neg)) via the rank-sum statistic."""
    pos = np.asarray(pos, np.float64)
    neg = np.asarray(neg, np.float64)
    n_p, n_n = len(pos), len(neg)
    if n_p == 0 or n_n == 0:
        return float("nan")
    ranks = _rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[:n_p].sum()
    return float((r_pos - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def value_discrimination(df, value_col: str = "V", group_cols=("task",)):
    """Per region: AUC of `value_col` separating success_bound (positive) from
    fail_bound, plus mean dV after per-group z-scoring. `df` has columns
    region, outcome, <value_col>, + group_cols."""
    import pandas as pd

    group_cols = list(group_cols)
    z = df.copy()
    z["_vz"] = z.groupby(group_cols)[value_col].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9))
    rows = []
    for region, g in z.groupby("region"):
        pos = g[g["outcome"] == "success_bound"]
        neg = g[g["outcome"] == "fail_bound"]
        rows.append({"region": region,
                     "auc": _auc(pos[value_col].to_numpy(),
                                 neg[value_col].to_numpy()),
                     "mean_dV": float(pos["_vz"].mean() - neg["_vz"].mean())
                     if len(pos) and len(neg) else float("nan"),
                     "n_success": int(len(pos)), "n_fail": int(len(neg))})
    return pd.DataFrame(rows)
