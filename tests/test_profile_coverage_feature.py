import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals._profile_core import probe_coverage
from evals.phase_probe import Thresholds


def _cube_ep(cube_xyz, grip, tmask, outcome="success"):
    return {"cube": np.asarray(cube_xyz, float).reshape(-1, 3),
            "grip": np.asarray(grip, float),
            "transport_mask": np.asarray(tmask, bool),
            "table_z": 0.02, "outcome": outcome}


def test_coverage_feature_cube_nn_distance():
    ref = np.array([[0, 0, 0], [10, 10, 10]], float)
    ep = _cube_ep([[0, 0, 0]], grip=[0.0], tmask=[False])
    df = probe_coverage([ep], ref, Thresholds(), feature="cube")
    assert set(df.columns) >= {"outcome", "region", "nn_dist"}
    assert df["region"].iloc[0] == "reach"
    assert abs(float(df["nn_dist"].iloc[0])) < 1e-6   # sits on a ref point


def test_coverage_feature_cube_needs_no_obs_key():
    ref = np.array([[0, 0, 0.02], [0.4, 0.1, 0.2]], float)
    ep = _cube_ep([[0.4, 0.1, 0.2], [0.0, 0.0, 0.02]],
                  grip=[1.0, 1.0], tmask=[True, False])
    df = probe_coverage([ep], ref, Thresholds(), feature="cube")
    assert len(df) == 2
    assert "obs" not in ep


def test_coverage_feature_obs_unchanged():
    ref = np.zeros((10, 4), float)
    ep = {"obs": np.array([[0, 0, 0, 0], [9, 9, 9, 9]], float),
          "grip": [0.0, 1.0], "transport_mask": [False, True],
          "cube": [[0, 0, 0.02], [0, 0, 0.2]],
          "table_z": 0.02, "outcome": "success"}
    df = probe_coverage([ep], ref, Thresholds())  # default feature="obs"
    assert df["nn_dist"].iloc[1] > df["nn_dist"].iloc[0]
