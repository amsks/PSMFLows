import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals import dataset_support as ds


def _make_buffer(tmp_path, cube_xy, n_ep=3):
    """Write npz episodes whose physics[:, 7:9] == cube_xy, plus
    decoy observation columns that do NOT match cube_xy."""
    buf = tmp_path / "buffer"
    buf.mkdir()
    T = len(cube_xy)
    for e in range(n_ep):
        phys = np.zeros((T, 21), np.float32)
        phys[:, 7:9] = cube_xy
        obs = np.random.RandomState(e).randn(T, 28).astype(np.float32)
        np.savez(buf / f"episode_{e:06d}_{T}.npz",
                 observation=obs, physics=phys,
                 action=np.zeros((T, 5), np.float32),
                 reward=np.zeros((T, 1), np.float32),
                 discount=np.ones((T, 1), np.float32))
    return buf


def test_decode_finds_matching_slice(tmp_path):
    # Real scenario: dataset cube xy and rollout cube xy are large,
    # independent samples over the SAME workspace distribution.
    rng = np.random.RandomState(0)
    buf_xy = rng.uniform(-0.3, 0.5, size=(2500, 2))
    ref = rng.uniform(-0.3, 0.5, size=(2500, 2))   # independent draw
    buf = _make_buffer(tmp_path, buf_xy)
    xy = ds.dataset_cube_xy(buf, ref_cube_xy=ref, n_files=10)
    assert xy.shape[1] == 2
    # decoded slice is the physics window, not a decoy obs column
    assert xy[:, 0].min() >= buf_xy[:, 0].min() - 1e-6
    assert xy[:, 0].max() <= buf_xy[:, 0].max() + 1e-6


def test_decode_raises_when_no_slice_matches(tmp_path):
    ref = np.full((100, 2), 999.0)          # nothing in buffer matches
    buf = _make_buffer(tmp_path, np.zeros((50, 2)))
    with pytest.raises(ValueError, match="no cube-xy slice"):
        ds.dataset_cube_xy(buf, ref_cube_xy=ref, n_files=10)


def test_support_kde_shape_and_norm():
    rng = np.random.RandomState(1)
    xy = rng.normal(0, 0.1, size=(500, 2))
    gx = np.linspace(-0.5, 0.5, 20)
    gy = np.linspace(-0.5, 0.5, 15)
    k = ds.support_kde(xy, gx, gy)
    assert k.shape == (15, 20)
    assert 0.0 <= k.min() and abs(k.max() - 1.0) < 1e-9


def test_support_kde_histogram_fallback_on_degenerate():
    xy = np.tile([0.1, 0.2], (50, 1))       # singular covariance
    gx = np.linspace(0, 0.3, 8)
    gy = np.linspace(0, 0.3, 8)
    k = ds.support_kde(xy, gx, gy)
    assert k.shape == (8, 8)
    assert k.max() == 1.0
