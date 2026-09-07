"""CPU unit tests for tools/diag_ensemble_disagreement.py's pure half (binning, Spearman).

The GPU half restores a Stage-C checkpoint and the frozen flow, so it is not exercised
here; the binning and rank-correlation helpers are where a silent bug would land.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.diag_ensemble_disagreement import bin_by_quantiles, spearman, summarize_condition


def test_spearman_monotone_reversed_and_ties():
    x = np.arange(50, dtype=float)
    assert spearman(x, np.exp(x / 10)) > 0.999
    assert spearman(x, -x) < -0.999
    # tie-averaged ranks: a constant side is undefined, not +-1
    assert np.isnan(spearman(x, np.ones_like(x)))
    # an independent permutation reads near zero
    assert abs(spearman(x, np.random.default_rng(0).permutation(x))) < 0.4
    # ties are averaged: duplicate x with a monotone y still gives a positive but < 1 value
    xt = np.repeat(np.arange(10, dtype=float), 5)
    r = spearman(xt, np.arange(50, dtype=float))
    assert 0.9 < r < 1.0


def test_bin_by_quantiles_equal_counts_and_means():
    rng = np.random.default_rng(1)
    x = rng.normal(size=1000)
    y = 3.0 * x + 1.0
    bins = bin_by_quantiles(x, y, n_bins=5)
    assert len(bins) == 5
    assert sum(b['n'] for b in bins) == 1000
    assert all(abs(b['n'] - 200) <= 1 for b in bins)
    means = [b['mean'] for b in bins]
    assert means == sorted(means)
    assert bins[0]['lo'] == float(x.min()) and bins[-1]['hi'] == float(x.max())
    # every row falls in exactly one bin (max is in the last, closed bin)
    assert bins[-1]['n'] > 0


def test_summarize_condition_keys_and_signs():
    rng = np.random.default_rng(2)
    n = 600
    dist = rng.uniform(0.1, 3.0, size=n)
    rows = {
        'rel_disagreement': 0.5 * dist + 0.01 * rng.normal(size=n),
        'rel_diag': 0.2 * dist,
        'abs_disagreement': np.ones(n), 'abs_target': 2 * np.ones(n),
        'norm_index': rng.uniform(1, 5, size=n), 'norm_udata': rng.uniform(1, 5, size=n),
        'dist_to_data': dist,
    }
    s = summarize_condition(rows)
    assert s['n_rows'] == n
    assert s['spearman']['rel_vs_dist_to_data'] > 0.99
    assert s['spearman']['diag_vs_dist_to_data'] > 0.999
    assert abs(s['spearman']['rel_vs_norm_index']) < 0.2
    m = [b['mean'] for b in s['by_dist_to_data']]
    assert m == sorted(m) and len(m) == 5
