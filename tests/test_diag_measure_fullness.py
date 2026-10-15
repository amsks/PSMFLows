"""CPU tests for the pure half of tools/diag_measure_fullness.py: nearest-neighbour matching,
the discounted occupancy over a reference set, the measure-to-distribution read and the
per-probe comparison metrics, on hand-built vectors with known answers."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.diag_measure_fullness import (
    compare,
    entropy,
    measure_to_dist,
    nearest,
    obs_std,
    occupancy,
    parse_int_list,
    split_half_tv,
    summarise,
    tv,
)


def test_nearest_standardized_and_column_subset():
    X = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 1.0]])
    std = np.array([10.0, 1.0])           # column 0 is 10x wider: (10, 0) is one std unit from (0, 0)
    X_std = X / std
    # (6, 0.9): standardized (0.6, 0.9) -> nearest is (0, 1) at distance sqrt(0.36 + 0.01)
    assert nearest(X_std, np.array([6.0, 0.9]), std).tolist() == [2]
    # on column 0 only the same query is nearest to (10, 0) (0.6 vs 0.4)
    assert nearest(X_std, np.array([6.0, 0.9]), std, slice(0, 1)).tolist() == [1]
    assert nearest(X_std, np.array([[0.1, 0.0], [9.0, 0.0]]), std).tolist() == [0, 1]


def test_obs_std_constant_dimension_is_one():
    s = obs_std(np.array([[1.0, 0.0], [3.0, 0.0]]))
    np.testing.assert_allclose(s, [1.0, 1.0])


def test_occupancy_known_trajectory():
    gamma = 0.5
    # one rollout visiting x=2, x=0, x=0: weights 1, .5, .25 over 1.75
    rho = occupancy([np.array([2, 0, 0])], n_ref=3, gamma=gamma)
    np.testing.assert_allclose(rho, [0.75 / 1.75, 0.0, 1.0 / 1.75])
    assert abs(rho.sum() - 1.0) < 1e-12
    # a second rollout of one step at x=1 is averaged in with weight 1/2
    rho2 = occupancy([np.array([2, 0, 0]), np.array([1])], n_ref=3, gamma=gamma)
    np.testing.assert_allclose(rho2, [0.5 * 0.75 / 1.75, 0.5, 0.5 / 1.75])
    # an empty rollout contributes nothing but still counts in the average
    rho3 = occupancy([np.array([1]), np.array([], np.int64)], n_ref=2, gamma=gamma)
    np.testing.assert_allclose(rho3, [0.0, 0.5])


def test_measure_to_dist():
    m = np.array([0.0, np.log(3.0)])
    np.testing.assert_allclose(measure_to_dist(m, 'softmax', 1.0), [0.25, 0.75])
    np.testing.assert_allclose(measure_to_dist(np.array([0.0, 2 * np.log(3.0)]), 'softmax', 2.0), [0.25, 0.75])
    np.testing.assert_allclose(measure_to_dist(np.array([-1.0, 1.0, 3.0]), 'squared', 1.0), [0.0, 0.25, 0.75])
    np.testing.assert_allclose(measure_to_dist(np.array([-1.0, -2.0]), 'squared', 1.0), [0.5, 0.5])


def test_tv_entropy_and_compare():
    p = np.array([0.7, 0.2, 0.1, 0.0])
    rho = np.array([0.1, 0.5, 0.0, 0.4])
    assert abs(tv(p, rho) - 0.5 * (0.6 + 0.3 + 0.1 + 0.4)) < 1e-12
    assert abs(entropy(np.array([0.5, 0.5, 0.0])) - np.log(2.0)) < 1e-12
    m = compare(p, rho, next_idx=0)
    assert m['p_top1_idx'] == 0 and m['top1_is_next'] == 1.0
    assert abs(m['p_top1'] - 0.7) < 1e-12 and abs(m['p_next'] - 0.7) < 1e-12 and abs(m['rho_next'] - 0.1) < 1e-12
    # shared mass beyond the next state: min(0.2, 0.5) + min(0.1, 0) + min(0, 0.4) = 0.2
    assert abs(m['mass_beyond_next'] - 0.2) < 1e-12
    assert m['rho_support'] == 3
    assert abs(m['tv_uniform'] - tv(np.full(4, 0.25), rho)) < 1e-12
    assert 'tv_b' not in m
    # a rollout that lands where p points: next_idx = 1, p's top is 0 -> not the detector signature
    m1 = compare(p, rho, next_idx=1)
    assert m1['top1_is_next'] == 0.0 and abs(m1['mass_beyond_next'] - (0.1 + 0.0 + 0.0)) < 1e-12
    # critic B at the same probe
    mb = compare(p, rho, next_idx=0, p_b=rho)
    assert abs(mb['tv_b']) < 1e-12 and abs(mb['tv_a_b'] - tv(p, rho)) < 1e-12
    assert abs(mb['p_b_next'] - 0.1) < 1e-12 and mb['top1_b_is_next'] == 0.0
    # perfectly ranked distributions: spearman 1 over all and over the support
    q = np.array([0.4, 0.3, 0.2, 0.1])
    mq = compare(q, q, next_idx=0)
    assert abs(mq['spearman'] - 1.0) < 1e-12 and abs(mq['spearman_support'] - 1.0) < 1e-12


def test_split_half_tv():
    gamma = 0.9
    same = [np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1])]
    assert abs(split_half_tv(same, 2, gamma)) < 1e-12
    diff = [np.array([0]), np.array([0]), np.array([1]), np.array([1])]
    assert abs(split_half_tv(diff, 2, gamma) - 1.0) < 1e-12
    assert np.isnan(split_half_tv([np.array([0])], 2, gamma))
    # an odd rollout is dropped: halves are [0], [1]
    assert abs(split_half_tv([np.array([0]), np.array([1]), np.array([0])], 2, gamma) - 1.0) < 1e-12


def test_summarise_and_parse():
    recs = [{'tv': 0.2, 'H_p': float('nan')}, {'tv': 0.4, 'H_p': 1.0}]
    s = summarise(recs, keys=('tv', 'H_p', 'tv_b'))
    assert s['n'] == 2 and abs(s['tv']['mean'] - 0.3) < 1e-12 and s['tv']['n'] == 2
    assert abs(s['tv']['se'] - np.std([0.2, 0.4], ddof=1) / np.sqrt(2)) < 1e-12
    assert s['H_p']['n'] == 1 and s['H_p']['se'] == 0.0 and 'tv_b' not in s
    assert parse_int_list(None, (0, 20)) == [0, 20]
    assert parse_int_list('0,40', (0,)) == [0, 40]
    assert parse_int_list('[1 2 3]', (0,)) == [1, 2, 3]
    assert parse_int_list([4, 5], (0,)) == [4, 5]
