"""CPU unit tests for tools/diag_gpi_selection.py's pure half.

The GPU half needs a restored Stage-C checkpoint, the frozen flow and a mujoco env, so it
is not exercised here (there is no cheap dummy: `psmflow.create` loads the flow ckpt).
What IS testable without any of that is the part that turns four checkpoints' Q tensors
into the drift table -- and that is the part whose bugs would be silent.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.diag_gpi_selection import _q, _spearman, _topk_overlap, compare


def test_spearman_monotone_and_reversed():
    x = np.arange(20, dtype=float)
    assert _spearman(x, 2 * x + 1) > 0.999
    assert _spearman(x, -x) < -0.999
    # ties are not averaged (argsort-of-argsort); Q values are continuous, so this is a
    # documented assumption rather than a gap -- an independent permutation reads ~0.
    assert abs(_spearman(x, np.random.default_rng(3).permutation(x))) < 0.6


def test_topk_overlap():
    a = np.arange(10, dtype=float)
    assert _topk_overlap(a, a, 5) == 1.0
    assert _topk_overlap(a, -a, 5) == 0.0


def test_q_summary_keys():
    s = _q(np.array([0.0, 1.0, np.nan, 2.0]))
    assert set(s) == {'mean', 'std', 'p10', 'p50', 'p90'} and abs(s['mean'] - 1.0) < 1e-9


def test_compare_identical_and_independent(tmp_path):
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(8, 6, 6)).astype(np.float32)
    paths = []
    for ep, arr in ((100, Q), (200, Q.copy()), (300, rng.normal(size=(8, 6, 6)).astype(np.float32))):
        p = tmp_path / f'{ep}.npz'
        np.savez_compressed(p, restore_epoch=ep, fixed_Q=arr)
        paths.append(str(p))
    out = tmp_path / 'cmp.json'
    rep = compare(paths, str(out))
    with open(out) as fh:
        assert json.load(fh) == rep
    assert rep['pairs']['100_vs_200']['spearman_pairs_mean'] > 0.999
    assert rep['pairs']['100_vs_200']['argmax_u_agreement'] == 1.0
    assert abs(rep['pairs']['100_vs_300']['spearman_pairs_mean']) < 0.5
