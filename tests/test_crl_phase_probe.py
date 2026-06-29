"""Schema and signal-extraction contract tests for the CRL phase probe.

Spec: docs/superpowers/specs/2026-05-27-crl-phase-failure-analysis-design.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FB_PARQUET = REPO / "analysis/legacy/phase_probe/aggregate/aggregate_per_episode.parquet"
CRL_AGG = REPO / "analysis/probes/phase_probe_crl/aggregate/aggregate_per_episode.parquet"


# Columns actually written by the existing FB pipeline (verified by reading
# analysis/legacy/phase_probe/aggregate/aggregate_per_episode.parquet). The aggregate
# adds `seed`; per-seed parquets don't carry it.
EXPECTED_COLUMNS = {
    "task", "scenario", "episode",
    "reached", "secured", "success",
    "reached_raw", "secured_raw",
    "furthest_phase", "fail_phase",
    "min_eff_cube_dist", "max_cube_lift",
    "final_cube_goal_dist", "length",
}


def _schema(parquet: Path) -> dict[str, str]:
    df = pd.read_parquet(parquet)
    return {c: str(df[c].dtype) for c in df.columns}


def test_fb_parquet_has_expected_columns():
    if not FB_PARQUET.exists():
        pytest.skip(f"FB reference parquet missing: {FB_PARQUET}")
    have = set(_schema(FB_PARQUET))
    missing = EXPECTED_COLUMNS - have
    assert not missing, (
        f"FB parquet missing columns expected by spec: {missing}; "
        f"got {sorted(have)}"
    )


def test_crl_parquet_is_fb_superset():
    """CRL parquet must contain every FB column with matching dtype.
    CRL may have extra columns (e.g. final_cube_lift) that the FB
    aggregator dropped — those don't break downstream consumers, which
    only read FB-named columns."""
    if not (FB_PARQUET.exists() and CRL_AGG.exists()):
        pytest.skip("FB or CRL parquet missing (run phase.sh first)")
    fb = _schema(FB_PARQUET)
    crl = _schema(CRL_AGG)
    missing = set(fb) - set(crl)
    assert not missing, (
        f"CRL parquet missing FB columns: {missing}; got {sorted(crl)}"
    )
    bad_dtype = [c for c in fb if fb[c] != crl[c]]
    assert not bad_dtype, (
        "CRL dtypes differ from FB for shared columns: "
        + ", ".join(f"{c} (FB={fb[c]} CRL={crl[c]})" for c in bad_dtype)
    )


# Synthetic-trajectory tests using the real classify_phases logic
# from evals.phase_probe — locks the signal-dict contract so the CRL
# rollout loop (Task 7) is guaranteed to produce the right keys.


def _make_signals(eff, cube, grip, goal, success=False, table_z=0.0):
    return {
        "eff": np.asarray(eff, dtype=np.float64),
        "cube": np.asarray(cube, dtype=np.float64),
        "grip": np.asarray(grip, dtype=np.float64),
        "goal": np.asarray(goal, dtype=np.float64),
        "success": bool(success),
        "table_z": float(table_z),
        "length": len(np.asarray(eff)),
    }


def test_classify_success_trajectory():
    """A trajectory that approaches, grasps, lifts, and arrives at the goal
    should classify as furthest_phase='success' / fail_phase='ok'."""
    from evals.phase_probe import Thresholds, classify_phases

    T = 200
    # Approach: gripper moves from origin to within 4 cm of cube at x=0.3
    eff = np.zeros((T, 3))
    cube = np.zeros((T, 3))
    cube[:, 0] = 0.30
    cube[:, 2] = 0.0
    goal = np.array([0.10, 0.0, 0.05])
    grip = np.ones(T)  # > tau_grip=0.5 means "closed"

    # t=0..50: approach
    eff[:50, 0] = np.linspace(0.00, 0.28, 50)
    # t=50..200: cube lifted in gripper, both move to goal
    eff[50:, 0] = np.linspace(0.28, 0.10, 150)
    eff[50:, 2] = np.linspace(0.00, 0.05, 150)
    cube[50:, 0] = np.linspace(0.30, 0.10, 150)
    cube[50:, 2] = np.linspace(0.00, 0.05, 150)
    grip[:50] = 0.0  # open before grasp

    signals = _make_signals(eff, cube, grip, goal, success=True, table_z=0.0)
    out = classify_phases(signals, Thresholds())
    assert out["success"], out
    assert out["furthest_phase"] == "success", out
    # fail_phase is "none" on success (verified by reading classify_phases).
    assert out["fail_phase"] == "none", out


def test_classify_approach_failure():
    """A trajectory where the gripper never gets close to the cube should
    classify as furthest_phase='none' / fail_phase='reach' (or similar)."""
    from evals.phase_probe import Thresholds, classify_phases

    T = 200
    eff = np.zeros((T, 3))           # gripper stuck at origin
    cube = np.full((T, 3), 0.50)     # cube ~70 cm away (far)
    cube[:, 2] = 0.0
    goal = np.array([0.50, 0.50, 0.05])
    grip = np.zeros(T)

    signals = _make_signals(eff, cube, grip, goal, success=False, table_z=0.0)
    out = classify_phases(signals, Thresholds())
    assert not out["success"], out
    assert out["furthest_phase"] == "none", out
    assert out["fail_phase"] == "reach", out


def test_classify_signal_dict_contract():
    """The signal dict produced by the rollout loop must accept the exact
    keys classify_phases requires. This locks the contract for Task 7."""
    from evals.phase_probe import Thresholds, classify_phases

    sig = _make_signals(
        eff=np.zeros((5, 3)), cube=np.full((5, 3), 1.0), grip=np.zeros(5),
        goal=np.zeros(3), success=False, table_z=0.0,
    )
    out = classify_phases(sig, Thresholds())
    # Every column in EXPECTED_COLUMNS that comes from classify_phases must
    # be in `out`.
    from_classifier = {
        "reached", "secured", "success",
        "reached_raw", "secured_raw",
        "furthest_phase", "fail_phase",
        "min_eff_cube_dist", "max_cube_lift",
        "final_cube_goal_dist", "length",
    }
    missing = from_classifier - set(out)
    assert not missing, (
        f"classify_phases output missing keys: {missing}; "
        f"got {sorted(out)}"
    )
