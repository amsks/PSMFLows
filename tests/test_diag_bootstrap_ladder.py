"""CPU tests for the pure numpy helpers of tools/diag_bootstrap_ladder.py: the percentile rank,
the best-of-2 pair picks, the kNN action-coverage helper, the self-rating block and the
ladder aggregation. No checkpoint, no jax."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.diag_bootstrap_ladder import (
    METRIC_GROUPS,
    best_of_pairs,
    knn_action_coverage,
    ladder_series,
    percentile_rank,
    self_rating_stats,
)


def test_percentile_rank_basic():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    r = percentile_rank(np.array([0.0, 2.5, 5.0, 2.0]), ref)
    assert np.allclose(r, [0.0, 0.5, 1.0, 0.375])        # ties count half


def test_percentile_rank_of_reference_itself_averages_half():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(64)
    assert abs(percentile_rank(ref, ref).mean() - 0.5) < 1e-12


def test_best_of_pairs_picks_argmax_within_disjoint_pairs():
    q = np.array([0.1, 0.9, 0.5, 0.2, 0.3, 0.3, 0.7])    # odd length: the last entry is dropped
    picks = best_of_pairs(q)
    assert picks.tolist() == [1, 2, 4]
    assert np.all(q[picks] >= q[picks ^ 1])


def test_knn_action_coverage_toy():
    # two clusters of dataset states; the actions of each cluster are far apart, so the
    # neighbours' actions decide the distance, not the global nearest action
    ds_obs = np.concatenate([np.zeros((10, 2)), np.full((10, 2), 10.0)])
    ds_act = np.concatenate([np.zeros((10, 1)), np.ones((10, 1))])
    ds_act[3] = 0.2                                                    # one off action in cluster 0
    obs_q = np.array([[0.1, 0.0], [9.9, 10.0]])
    acts_q = np.array([[[0.2], [1.0]], [[1.0], [0.0]]])                # (S=2, K=2, A=1)
    out = knn_action_coverage(obs_q, acts_q, ds_obs, ds_act, k=10, chunk=1)
    assert out.shape == (2, 2)
    assert np.allclose(out[0], [0.0, 0.8])                             # nearest action within cluster 0
    assert np.allclose(out[1], [0.0, 1.0])                             # cluster 1 has only action 1
    # k larger than the dataset is clipped; chunk larger than S is fine
    out_all = knn_action_coverage(obs_q, acts_q, ds_obs, ds_act, k=999, chunk=64)
    assert np.allclose(out_all[0], [0.0, 0.0])                         # every action is now a neighbour


def test_self_rating_stats_constant_offset():
    rng = np.random.default_rng(1)
    q_bc = rng.uniform(size=64)
    q_act = q_bc + 0.1
    out = self_rating_stats(q_bc, q_act)
    assert abs(out['dq_actor'] - 0.1) < 1e-12
    assert out['dq_bo64'] >= out['dq_bo2'] >= 0.0
    assert abs(out['q_bo64'] - q_bc.max()) < 1e-12
    assert out['rank_actor'] > 0.5
    zero = self_rating_stats(q_bc, q_bc)
    assert abs(zero['dq_actor']) < 1e-12 and abs(zero['rank_actor'] - 0.5) < 1e-12


def test_ladder_series_sorts_steps_and_groups_by_tag():
    reports = [
        {'tag': 'A', 'restore_epoch': 50000, 'metrics': {'x': 2.0, 'y': None}, 'metrics_sem': {'x': 0.2}},
        {'tag': 'A', 'restore_epoch': 25000, 'metrics': {'x': 1.0}, 'metrics_sem': {'x': 0.1}},
        {'tag': 'B', 'restore_epoch': 25000, 'metrics': {'x': 5.0}},
    ]
    s = ladder_series(reports)
    assert set(s) == {'A', 'B'}
    assert s['A']['x'] == {'steps': [25000, 50000], 'mean': [1.0, 2.0], 'sem': [0.1, 0.2]}
    assert 'y' not in s['A']
    assert s['B']['x'] == {'steps': [25000], 'mean': [5.0], 'sem': [0.0]}


def test_metric_groups_are_unique_names():
    names = [m for _, ms in METRIC_GROUPS for m in ms]
    assert len(names) == len(set(names))
