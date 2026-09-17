"""CPU unit tests for tools/diag_policy_family_diversity.py's pure half.

The rollout half needs the frozen flow, a restored affine checkpoint and a mujoco env, so
it is exercised by the CPU smoke in the tool's docstring, not here. What IS testable
without any of that is every number the table is built from: the MMD^2 estimator, its
floor/ceiling normalisation, the linear two-sample test and the coherence decomposition.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.diag_policy_family_diversity import (
    cv_softmax_accuracy,
    gaussian_mmd2,
    index_coherence,
    median_bandwidth,
    normalised_distinguishability,
    pairwise_mmd2,
    traj_features,
    upper_mean,
)


def test_mmd2_identical_is_zero_and_shift_is_positive():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    bw = median_bandwidth(X, rng)
    b, u = gaussian_mmd2(X, X.copy(), bw)
    assert abs(b) < 1e-9          # the V-statistic is exactly 0 for identical clouds
    assert -0.01 < u <= 0.0       # the U-statistic is slightly negative there (diagonal removed)
    b2, u2 = gaussian_mmd2(X, X + 2.0, bw)
    assert b2 > 0.05 and u2 > 0.05
    # same distribution, different draw: the biased form carries an O(1/m) positive
    # bias, the unbiased one is near 0 -- and both are far below the shifted value
    b3, u3 = gaussian_mmd2(X, rng.normal(size=(300, 5)), bw)
    assert 0 < b3 < b2 and abs(u3) < b2 / 10


def test_pairwise_matrix_symmetric_with_zero_diagonal():
    rng = np.random.default_rng(1)
    clouds = [rng.normal(size=(100, 3)) + k for k in range(3)]
    B, _U, selfm = pairwise_mmd2(clouds, 1.0)
    assert np.allclose(B, B.T) and np.allclose(np.diag(B), 0) and len(selfm) == 3
    assert B[0, 2] > B[0, 1] > 0
    assert upper_mean(B) == np.mean([B[0, 1], B[0, 2], B[1, 2]])
    assert upper_mean(np.zeros((1, 1))) is None


def test_normalisation_clips_to_unit_interval():
    assert normalised_distinguishability(0.5, 0.0, 1.0) == 0.5
    assert normalised_distinguishability(-0.2, 0.0, 1.0) == 0.0
    assert normalised_distinguishability(3.0, 0.0, 1.0) == 1.0
    assert normalised_distinguishability(0.5, 1.0, 1.0) is None
    assert normalised_distinguishability(None, 0.0, 1.0) is None


def test_classifier_chance_on_random_labels_and_perfect_on_separable():
    rng = np.random.default_rng(2)
    K, n = 8, 40
    X = rng.normal(size=(K * n, 12))
    y = rng.integers(0, K, size=K * n)
    acc, chance = cv_softmax_accuracy(X, y, seed=0)
    assert chance == 1.0 / K
    assert abs(acc - chance) < 0.08
    y2 = np.repeat(np.arange(K), n)
    X2 = rng.normal(size=(K * n, 12)) * 0.1 + 3.0 * rng.normal(size=(K, 12))[y2]
    acc2, _ = cv_softmax_accuracy(X2, y2, seed=0)
    assert acc2 > 0.95
    # too few trajectories for any fold to have a training set: None, not a crash
    assert cv_softmax_accuracy(X[:2], np.array([0, 1]))[0] is None


def test_index_coherence_consistent_vs_independent():
    rng = np.random.default_rng(3)
    S, K, d = 200, 8, 4
    offsets = rng.normal(size=(K, d))
    consistent = rng.normal(size=(S, 1, d)) + offsets[None] + 0.1 * rng.normal(size=(S, K, d))
    c = index_coherence(consistent, rng)
    assert c['frac_index'] > 0.9 and c['frac_index_shuffled'] < 0.1
    assert len(c['frac_index_per_dim']) == d
    independent = rng.normal(size=(S, K, d))
    i = index_coherence(independent, rng)
    assert i['frac_index'] < 0.05 and abs(i['frac_index'] - i['frac_index_shuffled']) < 0.05


def test_traj_features_shape():
    F = traj_features([np.ones((10, 3)), np.arange(6.0).reshape(2, 3)])
    assert F.shape == (2, 6) and np.allclose(F[0], [1, 1, 1, 0, 0, 0])
