"""CPU tests for the pure helpers of tools/diag_action_from_future.py (no jax, no dataset)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.diag_action_from_future import (
    BUCKETS,
    bucket_shares,
    eligible_rows,
    geometric_bucket_shares,
    geometric_k,
    split_trajectories,
    summarise_scores,
    trajectory_ends,
)


def test_trajectory_ends_and_unmarked_last_row():
    terminals = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0], np.float32)   # last trajectory not marked
    assert trajectory_ends(terminals).tolist() == [2, 2, 2, 6, 6, 6, 6, 8, 8]


def test_split_keeps_trajectories_whole():
    end_of_row = trajectory_ends(np.tile(np.r_[np.zeros(9), 1.0], 50))   # 50 trajectories of 10 rows
    traj_id, held = split_trajectories(end_of_row, 0.1, seed=0)
    assert traj_id.max() == 49 and held.sum() == 50
    per_traj = np.bincount(traj_id, weights=held)
    assert set(per_traj.tolist()) == {0.0, 10.0}
    _, held2 = split_trajectories(end_of_row, 0.1, seed=0)
    assert np.array_equal(held, held2)


def test_geometric_draws_match_analytic_bucket_shares():
    analytic = geometric_bucket_shares(0.98)
    assert abs(sum(analytic.values()) - 1.0) < 1e-12
    assert abs(analytic['<=1'] - 0.02) < 1e-12
    assert abs(analytic['>50'] - 0.98 ** 50) < 1e-12
    k = geometric_k(1.0 - np.random.default_rng(0).random(400000), 0.98)
    assert k.min() == 1
    empirical = bucket_shares(k)
    for label, _, _ in BUCKETS:
        assert abs(empirical[label] - analytic[label]) < 0.004


def test_eligible_rows_need_k_steps_left():
    end_of_row = trajectory_ends(np.array([0, 0, 0, 1, 0, 0, 1], np.float32))
    rows = np.arange(7)
    assert eligible_rows(rows, end_of_row, 1).tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert eligible_rows(rows, end_of_row, 3).tolist() == [0, 1, 4]
    assert eligible_rows(rows, end_of_row, 5).tolist() == []


def test_summarise_scores_chance_perfect_and_ties():
    rng = np.random.default_rng(0)
    n, c = 20000, 64
    cluster = np.repeat(np.arange(100), n // 100)
    st = summarise_scores(rng.normal(size=(n, c)) * 1e-3, cluster)
    assert abs(st['top1_accuracy'] - 1 / c) < 4 * st['top1_se_binomial']
    assert abs(st['information_nats']) < 1e-3
    assert 0.5 < st['top1_se'] / st['top1_se_binomial'] < 2.0     # independent rows: both agree
    scores = rng.normal(size=(n, c))
    scores[:, 0] = 50.0
    st = summarise_scores(scores, cluster)
    assert st['top1_accuracy'] == 1.0
    assert abs(st['information_nats'] - np.log(c)) < 1e-6
    st = summarise_scores(np.zeros((n, c)), cluster)                # constant score
    assert st['top1_accuracy'] == 0.0 and st['score_spread'] == 0.0
    assert abs(st['information_nats']) < 1e-12


def test_cluster_se_exceeds_binomial_when_rows_of_a_trajectory_agree():
    rng = np.random.default_rng(1)
    c = 8
    hit = rng.random(50) < 0.5                                       # whole trajectory hit or miss
    scores = np.zeros((50 * 100, c))
    scores[:, 0] = np.repeat(np.where(hit, 1.0, -1.0), 100)
    st = summarise_scores(scores, np.repeat(np.arange(50), 100))
    assert st['top1_se'] > 5 * st['top1_se_binomial']
