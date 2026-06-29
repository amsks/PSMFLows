import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.training_value import (region_labels, cube_to_goal_dist,
                                   phase_spearman_table, horizon_reach_label,
                                   outcome_spearman_table, value_discrimination,
                                   flow_step_labels)
from evals.phase_probe import Thresholds


def test_region_labels_reach_grasp_transport():
    thr = Thresholds()
    grip = np.array([0.0, 0.9, 0.9])          # open, closed, closed
    lift = np.array([0.0, 0.0, 0.20])         # on table, on table, lifted
    out = region_labels(grip, lift, thr)
    assert list(out) == ["reach", "grasp", "transport"]


def test_cube_to_goal_dist():
    cube = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    d = cube_to_goal_dist(cube, np.array([0.0, 0.0, 0.0]))
    assert np.allclose(d, [0.0, 5.0])


def test_phase_spearman_table_monotone():
    rows = []
    for ph in ("reach", "transport"):
        for i in range(8):
            d = 0.3 - 0.03 * i
            rows.append({"task": "task1", "region": ph, "d": d,
                         "V": 1.0 - d})
    df = pd.DataFrame(rows)
    tab = phase_spearman_table(df, value_col="V")
    r = tab.set_index("region")["rho_mean"]
    assert r.loc["transport"] > 0.99 and r.loc["reach"] > 0.99


def test_horizon_reach_label_window_and_thresh():
    d = np.array([0.10, 0.08, 0.03, 0.20])
    assert horizon_reach_label(d, 0.04, 2) is False   # min(first 2)=0.08
    assert horizon_reach_label(d, 0.04, 3) is True    # 0.03 < 0.04
    assert horizon_reach_label(np.array([0.05, 0.06]), 0.04, 5) is False
    assert horizon_reach_label(np.array([]), 0.04, 5) is False  # empty tail


def test_outcome_spearman_table_monotone():
    rows = []
    for oc in ("success_bound", "fail_bound"):
        for i in range(8):
            d = 0.3 - 0.03 * i
            rows.append({"task": "task1", "region": "transport",
                         "outcome": oc, "d": d, "V": 1.0 - d})
    tab = outcome_spearman_table(pd.DataFrame(rows), value_col="V")
    tab = tab.set_index(["region", "outcome"])["rho_mean"]
    assert tab.loc[("transport", "success_bound")] > 0.99
    assert tab.loc[("transport", "fail_bound")] > 0.99


def test_value_discrimination_separable_and_random():
    rows = []
    for t in ("task1", "task2"):
        for v in (1.0, 2.0, 3.0, 4.0):
            rows.append({"task": t, "region": "transport",
                         "outcome": "success_bound", "d": 0.1, "V": v + 10})
            rows.append({"task": t, "region": "transport",
                         "outcome": "fail_bound", "d": 0.1, "V": v})
    disc = value_discrimination(pd.DataFrame(rows), "V").set_index("region")
    assert disc.loc["transport", "auc"] > 0.99
    assert disc.loc["transport", "mean_dV"] > 0

    rng = np.random.default_rng(0)
    rrows = []
    for oc in ("success_bound", "fail_bound"):
        for v in rng.normal(size=2000):
            rrows.append({"task": "task1", "region": "reach",
                          "outcome": oc, "d": 0.1, "V": float(v)})
    auc = value_discrimination(pd.DataFrame(rrows), "V").set_index(
        "region").loc["reach", "auc"]
    assert abs(auc - 0.5) < 0.05


def test_flow_step_labels_bounded_reach():
    # one episode, cube first reaches goal g (0.30) at step 3, then leaves
    cube = np.array([[0.50, 0.0, 0.0],   # d=0.20
                     [0.45, 0.0, 0.0],   # d=0.15
                     [0.40, 0.0, 0.0],   # d=0.10
                     [0.30, 0.0, 0.0],   # d=0.00 -> reaches
                     [0.60, 0.0, 0.0]])  # d=0.30
    g = np.array([[0.30, 0.0, 0.0]])     # one goal
    out = flow_step_labels(cube, g, horizon=2, thresh=0.04)  # [T, n_goals]
    # horizon=2: step2 tail {0.10,0.00} hits; step3 tail {0.00,0.30} hits
    assert out.shape == (5, 1)
    assert out[2, 0] and out[3, 0]
    assert not out[0, 0] and not out[1, 0]
    assert not out[4, 0]                 # last step, tail {0.30}, no reach
