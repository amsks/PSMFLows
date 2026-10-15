"""CPU tests for the pure numpy metrics of tools/diag_actor_mechanism.py on synthetic samples."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.diag_actor_mechanism import (
    bimodality,
    chi2_ball_reference,
    cosine_rows,
    diff_stats,
    excess_kurtosis_per_dim,
    gaussian_fit,
    gaussian_loglik,
    mean_explained_fraction,
    nn_membership,
    nn_membership_loo,
    prior_loglik,
    shape_stats,
    within_std_ball_fraction,
)


def test_bimodality_two_clusters_vs_one_blob():
    rng = np.random.default_rng(0)
    one = rng.standard_normal((400, 5))
    two = np.concatenate([rng.standard_normal((200, 5)) + 6.0, rng.standard_normal((200, 5)) - 6.0])
    b1, b2 = bimodality(one, rng), bimodality(two, rng)
    assert b1['separation'] < 3.0
    assert b2['separation'] > 3.0
    assert abs(b2['minor_share'] - 0.5) < 0.05


def test_kurtosis_gaussian_near_zero_and_heavy_tails_positive():
    rng = np.random.default_rng(1)
    g = rng.standard_normal((200_000, 4))
    assert abs(excess_kurtosis_per_dim(g)) < 0.05
    heavy = rng.standard_t(df=4, size=(200_000, 4))
    assert excess_kurtosis_per_dim(heavy) > 1.0
    bimodal = np.concatenate([np.ones((1000, 1)), -np.ones((1000, 1))])
    assert excess_kurtosis_per_dim(bimodal) < -1.5


def test_mean_explained_fraction_constant_shift_is_one():
    rng = np.random.default_rng(2)
    const = np.tile(np.array([0.3, -0.2, 0.1]), (500, 1))
    assert abs(mean_explained_fraction(const) - 1.0) < 1e-12
    zero_mean = rng.standard_normal((20_000, 3))
    assert mean_explained_fraction(zero_mean) < 0.01
    assert mean_explained_fraction(np.zeros((10, 3))) == 0.0


def test_gaussian_loglik_matches_prior_for_standard_normal():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((50, 3))
    ll = gaussian_loglik(x, np.zeros(3), np.eye(3))
    np.testing.assert_allclose(ll, prior_loglik(x), rtol=1e-12)


def test_within_ball_matches_chi2_reference_and_shape_stats_keys():
    rng = np.random.default_rng(4)
    d = 5
    x = rng.standard_normal((50_000, d))
    mu, cov = gaussian_fit(x)
    frac = within_std_ball_fraction(x, mu, cov)
    ref = chi2_ball_reference(d, np.random.default_rng(5), n=100_000)
    assert abs(frac - ref) < 0.02
    st = shape_stats(x[:500], x, rng)
    for k in ('norm_mean_ratio', 'norm_std_ratio', 'kurtosis', 'loglik_fit_minus_prior',
              'within_std_ball', 'separation', 'minor_share', 'centroid_norm'):
        assert k in st
    assert abs(st['norm_mean_ratio'] - 1.0) < 0.1
    # a shell: norms concentrated, mean norm above the prior's
    shell = x[:500] / np.linalg.norm(x[:500], axis=1, keepdims=True) * 4.0
    sh = shape_stats(shell, x, rng)
    assert sh['norm_std_ratio'] < 0.05 and sh['norm_mean_ratio'] > 1.5


def test_cosine_rows_and_nn_membership():
    a = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    b = np.array([[2.0, 0.0], [0.0, -1.0], [1.0, 1.0]])
    np.testing.assert_allclose(cosine_rows(a, b), [1.0, -1.0, 1.0])
    ref = np.array([[0.0, 0.0], [10.0, 10.0], [10.0, 10.5], [0.0, 0.5]])
    member = np.array([True, False, False, True])
    assert nn_membership(np.array([[0.1, 0.1], [9.9, 10.1]]), ref, member) == 0.5
    assert nn_membership_loo(ref, member) == 0.5


def test_diff_stats():
    d = diff_stats(np.array([0.2, 0.3, 0.1]), np.array([0.1, 0.1, 0.1]))
    np.testing.assert_allclose(d['diff_mean'], 0.1)
    np.testing.assert_allclose(d['diff_rel'], 1.0)
    np.testing.assert_allclose(d['frac_pos'], 2.0 / 3.0)
