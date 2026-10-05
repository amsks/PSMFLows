"""CPU tests for the pure half of tools/diag_psmgoal_rollout_ranking.py: step bucketing and
the per-bucket ranking statistics, on synthetic score arrays."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.diag_psmgoal_rollout_ranking import aggregate, bucket_index, step_stats, summarize


def test_bucket_edges():
    t = np.array([0, 24, 25, 49, 50, 99, 100, 199, 200, 350])
    assert bucket_index(t).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_step_stats_known_values():
    q = np.array([[0.0, 1.0, 2.0, 3.0, 4.0],        # std sqrt(2), best-median 2, no tie
                  [1.0, 1.0, 1.0, 1.0, 1.0],        # zero spread, exact tie
                  [10.0, 9.95, 0.0, 0.0, 0.0]])     # top-2 gap 0.05 <= 1% of 10 -> tie
    st = step_stats(q)
    np.testing.assert_allclose(st["spread"], [np.sqrt(2.0), 0.0, np.std(q[2])])
    np.testing.assert_allclose(st["best_minus_median"], [2.0, 0.0, 10.0])
    assert st["near_tie"].tolist() == [False, True, True]
    np.testing.assert_allclose(st["chosen"], [4.0, 1.0, 10.0])


def test_summarize_splits_outcome_and_normalizes():
    rng = np.random.default_rng(0)
    # ep 0 succeeds with 30 steps, ep 1 fails with 210 steps; K = 8
    t = np.concatenate([np.arange(30), np.arange(210)])
    ep = np.concatenate([np.zeros(30, int), np.ones(210, int)])
    q = rng.normal(size=(t.size, 8))
    q[ep == 1] *= 3.0                                   # failed episode: 3x spread
    s = summarize(t, q, ep, np.array([True, False]), q_phiw=q, q_b=np.zeros_like(q))
    tab = s["table"]
    assert tab["success"]["0-24"]["n_steps"] == 25
    assert tab["success"]["25-49"]["n_steps"] == 5
    assert tab["success"]["50-99"]["n_steps"] == 0
    assert tab["fail"]["200+"]["n_steps"] == 10
    assert tab["fail"]["100-199"]["n_steps"] == 100
    chosen = q.max(1)
    sig = chosen.std()
    assert abs(s["sigma_chosen"] - sig) < 1e-12
    m = (ep == 1) & (t >= 50) & (t < 100)
    np.testing.assert_allclose(tab["fail"]["50-99"]["spread"], q[m].std(1).mean())
    np.testing.assert_allclose(tab["fail"]["50-99"]["spread_norm"], q[m].std(1).mean() / sig)
    np.testing.assert_allclose(tab["fail"]["50-99"]["chosen_z"], (chosen[m].mean() - chosen.mean()) / sig)
    # the scores ARE q_phiw here, so the argmax agrees at every step; q_b is flat
    assert tab["fail"]["50-99"]["argmax_eq_phiw_argmax"] == 1.0
    assert tab["fail"]["50-99"]["spread_b_norm"] == 0.0
    # failed-episode spread is ~3x the successful one
    assert tab["fail"]["0-24"]["spread"] > 2.0 * tab["success"]["0-24"]["spread"]
    # scale-free: multiplying every score by 100 leaves the *_norm fields unchanged
    s100 = summarize(t, 100 * q, ep, np.array([True, False]))
    np.testing.assert_allclose(s100["table"]["fail"]["0-24"]["spread_norm"],
                               tab["fail"]["0-24"]["spread_norm"])
    np.testing.assert_allclose(s100["table"]["success"]["25-49"]["best_minus_median_norm"],
                               tab["success"]["25-49"]["best_minus_median_norm"])
    assert s["success_rate"] == 0.5


def test_aggregate_averages_over_files_with_steps():
    t = np.arange(30)
    ep = np.zeros(30, int)
    q1 = np.tile(np.arange(4.0), (30, 1))
    a = summarize(t, q1, ep, np.array([True]))
    b = summarize(t, q1 * 2.0, ep, np.array([False]))
    agg = aggregate([a, b])
    # success cells come only from file a, fail cells only from file b
    assert agg["table"]["success"]["0-24"]["n_files"] == 1
    assert agg["table"]["fail"]["0-24"]["n_files"] == 1
    assert agg["table"]["success"]["0-24"]["spread"] == a["table"]["success"]["0-24"]["spread"]
    assert agg["table"]["success"]["50-99"]["n_files"] == 0
    assert agg["table"]["success"]["50-99"]["spread"] is None
    assert agg["success_rate_pooled"] == 0.5
