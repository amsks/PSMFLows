"""evals/_profile_core.py — torch-free probe math shared by the FB
(PyTorch) and GCIQL (JAX) analysis tracks. numpy/pandas only; never
import torch/jax here so it loads in either venv."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _spearman(y: np.ndarray, x: np.ndarray) -> float:
    """Spearman rho (Pearson on average ranks) of y vs x; nan if either
    side is constant or fewer than 2 points. Outlier-robust and
    scale-free, unlike an OLS slope on the unnormalised value V."""
    import pandas as pd

    y = np.asarray(y, np.float64)
    x = np.asarray(x, np.float64)
    if len(x) < 2 or np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def transport_mask(ep: Dict[str, Any], thr) -> np.ndarray:
    """Per-step boolean: cube lifted above table and gripper closed."""
    cube = np.asarray(ep["cube"], dtype=np.float64).reshape(-1, 3)
    grip = np.asarray(ep["grip"], dtype=np.float64).reshape(-1)
    lift = cube[:, 2] - float(ep["table_z"])
    return (lift > thr.delta_lift) & (grip > thr.tau_grip)


def _region(ep: Dict[str, Any], thr) -> np.ndarray:
    """Per-step region label: 'transport' if lifted+gripped, else
    'lift' if gripper closed, else 'reach'."""
    grip = np.asarray(ep["grip"], np.float64).reshape(-1)
    tm = np.asarray(ep["transport_mask"], bool)
    out = np.where(tm, "transport",
                   np.where(grip > thr.tau_grip, "lift", "reach"))
    return out


def _episode_feature(ep: Dict[str, Any], feature: str) -> np.ndarray:
    """Per-step feature matrix for coverage. 'obs' = raw observation
    (state path); 'cube' = cube xyz (physics-space, for pixels)."""
    if feature == "cube":
        return np.asarray(ep["cube"], np.float64).reshape(-1, 3)
    return np.asarray(ep["obs"], np.float64)


def probe_coverage(episodes, ref_obs: np.ndarray, thr,
                   max_ref: int = 5000, seed: int = 0,
                   feature: str = "obs") -> "Any":
    """Per-step nearest-neighbour L2 distance to the standardised offline
    reference, tagged by outcome and region. `feature` selects the space:
    'obs' (state observation, default) or 'cube' (physics-space cube xyz)."""
    import pandas as pd

    ref = np.asarray(ref_obs, np.float64)
    if len(ref) > max_ref:
        rng = np.random.default_rng(seed)
        ref = ref[rng.choice(len(ref), max_ref, replace=False)]
    mu = ref.mean(0)
    sd = ref.std(0) + 1e-6
    refn = (ref - mu) / sd
    rows = []
    for ep in episodes:
        feat = _episode_feature(ep, feature)
        if feat.shape[0] == 0:
            continue
        on = (feat - mu) / sd
        # batched: ||on||^2 + ||refn||^2 - 2 on·refn
        d2 = (on ** 2).sum(1, keepdims=True) + (refn ** 2).sum(1) \
            - 2.0 * on @ refn.T
        nn = np.sqrt(np.maximum(d2.min(1), 0.0))
        reg = _region(ep, thr)
        for r, n in zip(reg, nn):
            rows.append({"outcome": ep["outcome"], "region": str(r),
                         "nn_dist": float(n)})
    return pd.DataFrame(rows)


T1_RHO_MIN = 0.15
T1_RHO_GAP = 0.10
T2_SPARSE_FRAC = 0.05
T2_TOPK_AT_GOAL = 0.5
T3_R2_FAIL = 0.5
T3_ACC_FAIL = 0.75
T3_R2_RESOLVE = 0.7
T3_ACC_RESOLVE = 0.9
T4_OFFSUPPORT_RATIO = 1.25


def _isnan(v) -> bool:
    try:
        return bool(np.isnan(v))
    except TypeError:
        return True


def _verdict_t1(rho_succ, rho_fail):
    if _isnan(rho_succ) or _isnan(rho_fail):
        return "INSUFFICIENT DATA", "no valid transport-region episodes"
    if rho_succ >= T1_RHO_MIN and rho_succ - rho_fail >= T1_RHO_GAP:
        return "SUPPORTS", "success value rises toward goal; fail flatter"
    if rho_fail - rho_succ >= T1_RHO_GAP:
        return "CONTRADICTS", "fail value rises more than success"
    return "WEAK", "no clear success/fail gradient separation"


def _verdict_t2(frac, topk_at_goal):
    if _isnan(frac) or _isnan(topk_at_goal):
        return "INSUFFICIENT DATA", "no relabel/z-decoding stats"
    if frac < T2_SPARSE_FRAC and topk_at_goal < T2_TOPK_AT_GOAL:
        return "SUPPORTS", "data goal-sparse; z's top states off-goal"
    return "WEAK", "data not strongly goal-sparse or z points at goal"


def _verdict_t3(r2, acc):
    if _isnan(r2) or _isnan(acc):
        return "INSUFFICIENT DATA", "no B-resolution stats"
    if r2 < T3_R2_FAIL and acc < T3_ACC_FAIL:
        return "SUPPORTS", "B barely resolves placement distance"
    if r2 >= T3_R2_RESOLVE and acc >= T3_ACC_RESOLVE:
        return "RESOLVES", "B cleanly resolves placement (contradicts)"
    return "WEAK", "B partially resolves placement"


def _verdict_t4(fail_nn, succ_nn):
    if _isnan(fail_nn) or _isnan(succ_nn):
        return "INSUFFICIENT DATA", "missing transport-region coverage"
    if fail_nn >= T4_OFFSUPPORT_RATIO * succ_nn:
        return "SUPPORTS", "transport-fails sit further off data support"
    return "NEUTRAL", "fails not meaningfully more off-support"


def _synthesis(verdicts) -> str:
    supports = [k for k, (v, _) in verdicts.items() if v == "SUPPORTS"]
    contra = [k for k, (v, _) in verdicts.items()
              if v in ("CONTRADICTS", "RESOLVES")]
    if supports:
        s = (", ".join(supports)
             + " support the representation-failure hypothesis")
    else:
        s = "no probe supports the representation-failure hypothesis"
    if contra:
        s += "; " + ", ".join(contra) + " argue against it"
    return "Synthesis: " + s + "."
