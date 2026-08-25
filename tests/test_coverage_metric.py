"""Coverage: is a prior draw captured under any mixture component of a related state?

The metric has an exactly known value in one case, which is what makes it testable: a
mixture centred at 0 with covariance I has a 95% ellipsoid identical to the prior's 95%
ball, so coverage must be 0.95 and the misses must be exactly the top prior quantile.
"""
import numpy as np
import pytest

from tools.tune_preimage_inversion import _coverage

N, D = 256, 3


def _mixtures(mean, cov, n=N, d=D, k_comp=1):
    means = np.tile(np.asarray(mean, np.float32), (n, k_comp, 1))
    covs = np.tile(np.asarray(cov, np.float32), (n, k_comp, 1, 1))
    weights = np.full((n, k_comp), 1.0 / k_comp, np.float32)
    return means, covs, weights


def _obs(n=N, seed=0):
    return np.random.default_rng(seed).standard_normal((n, 4)).astype(np.float32)


def test_unit_covariance_covers_exactly_the_prior_95_percent_ball():
    means, covs, weights = _mixtures(np.zeros(D), np.eye(D))
    out = _coverage(means, covs, weights, _obs(), D, n_prior=16, k=8)
    assert out['covered_at_k']['0'] == pytest.approx(0.95, abs=0.02)


def test_misses_are_exactly_the_top_prior_quantile():
    """With cov = I the covered set is {||u||^2 <= chi2_95}, so the bucket boundaries line
    up with the metric's threshold: everything below the 95th percentile is in, the rest
    is out. This is what makes the radial breakdown readable -- a miss beyond u_clip is
    harmless, a miss in the bulk is not."""
    means, covs, weights = _mixtures(np.zeros(D), np.eye(D))
    by_q = _coverage(means, covs, weights, _obs(), D, n_prior=16, k=4)['covered_by_prior_quantile']
    for bucket in ('q0-25', 'q25-50', 'q50-75', 'q75-95'):
        assert by_q[bucket] == pytest.approx(1.0, abs=1e-9), bucket
    assert by_q['q95-100'] == pytest.approx(0.0, abs=1e-9)


def test_a_negligible_region_covers_nothing():
    means, covs, weights = _mixtures(np.zeros(D), 1e-6 * np.eye(D))
    out = _coverage(means, covs, weights, _obs(), D, n_prior=16, k=8)
    assert out['covered_at_k']['0'] == 0.0
    assert out['mean_count_knn'] == 0.0


def test_coverage_is_monotone_in_k():
    """More related states can only add coverage, never remove it."""
    rng = np.random.default_rng(1)
    means = rng.standard_normal((N, 1, D)).astype(np.float32)      # scattered regions
    covs = np.tile(0.25 * np.eye(D, dtype=np.float32), (N, 1, 1, 1))
    weights = np.ones((N, 1), np.float32)

    curve = _coverage(means, covs, weights, _obs(), D, n_prior=16, k=32)['covered_at_k']
    values = [curve[key] for key in sorted(curve, key=int)]
    assert values == sorted(values), curve
    assert values[-1] > values[0], 'scattered neighbours should add coverage over self alone'


def test_zero_weight_components_do_not_count():
    """A component the mixture never draws from is not part of the region it would cover."""
    means = np.zeros((N, 2, D), np.float32)
    covs = np.tile(np.eye(D, dtype=np.float32), (N, 2, 1, 1))
    covs[:, 0] *= 1e-6                       # live component: covers nothing
    weights = np.stack([np.ones(N), np.zeros(N)], 1).astype(np.float32)  # dead wide one

    out = _coverage(means, covs, weights, _obs(), D, n_prior=16, k=4)
    assert out['covered_at_k']['0'] == 0.0
