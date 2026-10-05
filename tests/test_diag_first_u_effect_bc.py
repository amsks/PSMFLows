"""CPU tests for the pure statistics of tools/diag_first_u_effect_bc.py (no env, no jax)."""
import os
import sys

import numpy as np
from scipy.stats import chi2_contingency

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.diag_first_u_effect_bc import ENVS, aggregate_anchors, candidate_effect_stats, default_out_name

K, R = 8, 100


def _draw(rates, seed):
    rng = np.random.default_rng(seed)
    success = (rng.random((len(rates), R)) < np.asarray(rates)[:, None]).astype(int)
    steps = rng.integers(1, 150, size=success.shape)
    return success, steps


def test_equal_rates_share_near_zero_and_large_p():
    success, steps = _draw([0.3] * K, seed=0)
    st = candidate_effect_stats(success, steps)
    assert st["chi2"]["dof"] == K - 1
    assert st["chi2"]["p"] > 0.05
    assert st["success"]["eta2"] < 0.03
    assert abs(st["success"]["omega2"]) < 0.02
    assert st["discounted"]["eta2"] < 0.03
    assert abs(st["discounted"]["omega2"]) < 0.02
    # observed between-candidate sd is at the sampling-noise level
    ratio = st["success"]["sd_between_observed"] / st["success"]["sd_between_expected_from_noise"]
    assert 0.4 < ratio < 1.8


def test_exactly_equal_counts_give_zero_share_and_p_one():
    success = np.zeros((K, R), int)
    success[:, :30] = 1                                  # every candidate: exactly 30 / 100
    st = candidate_effect_stats(success, np.full((K, R), 10))
    assert st["success"]["eta2"] == 0.0
    assert st["success"]["range"] == 0.0
    assert st["chi2"]["statistic"] == 0.0 and st["chi2"]["p"] == 1.0
    assert st["success"]["omega2"] < 0.0                 # bias-corrected share goes below zero
    np.testing.assert_allclose(st["success"]["sd_between_expected_from_noise"],
                               np.sqrt(0.3 * 0.7 / R))
    np.testing.assert_allclose(st["discounted"]["pooled_mean"], 0.3 * 0.98 ** 10)


def test_unequal_rates_large_share_and_tiny_p():
    success, steps = _draw(np.linspace(0.05, 0.9, K), seed=1)
    st = candidate_effect_stats(success, steps)
    assert st["chi2"]["p"] < 1e-10
    assert st["success"]["eta2"] > 0.2
    assert st["success"]["omega2"] > 0.2
    assert st["success"]["range"] > 0.6
    assert st["success"]["sd_between_observed"] > 3 * st["success"]["sd_between_expected_from_noise"]
    assert st["discounted"]["f_p"] < 1e-6


def test_chi2_matches_scipy_contingency_and_eta2_by_hand():
    success, steps = _draw([0.2, 0.25, 0.4, 0.1, 0.3, 0.35, 0.2, 0.5], seed=2)
    st = candidate_effect_stats(success, steps)
    k = success.sum(axis=1)
    ref = chi2_contingency(np.stack([k, R - k], axis=1), correction=False)
    np.testing.assert_allclose(st["chi2"]["statistic"], ref[0], rtol=1e-10)
    np.testing.assert_allclose(st["chi2"]["p"], ref[1], rtol=1e-10)
    y = success.astype(float)
    ss_b = R * ((y.mean(1) - y.mean()) ** 2).sum()
    ss_t = ((y - y.mean()) ** 2).sum()
    np.testing.assert_allclose(st["success"]["eta2"], ss_b / ss_t)
    ms_w = (ss_t - ss_b) / (K * (R - 1))
    np.testing.assert_allclose(st["success"]["omega2"], (ss_b - (K - 1) * ms_w) / (ss_t + ms_w))
    d = np.where(success == 1, 0.98 ** steps.astype(float), 0.0)
    np.testing.assert_allclose(st["discounted"]["per_candidate_mean"], d.mean(1))
    # for a 0/1 outcome, eta2 = chi2 / N
    np.testing.assert_allclose(st["success"]["eta2"], st["chi2"]["statistic"] / (K * R))


def test_no_success_is_undefined_not_zero():
    st = candidate_effect_stats(np.zeros((K, R), int), np.full((K, R), 100))
    assert st["chi2"]["p"] is None and st["chi2"]["statistic"] is None
    assert st["success"]["eta2"] is None and st["success"]["omega2"] is None
    assert st["success"]["pooled_mean"] == 0.0 and st["discounted"]["pooled_mean"] == 0.0


def test_aggregate_counts_and_means():
    anchors = []
    for task, t0, rates, seed in ((1, 0, [0.3] * K, 0), (1, 50, np.linspace(0.05, 0.9, K), 1),
                                  (2, 0, [0.0] * K, 3)):
        success, steps = _draw(rates, seed)
        anchors.append({"task": task, "prefix_len": t0, "stats": candidate_effect_stats(success, steps)})
    agg = aggregate_anchors(anchors)
    assert agg["n_anchors"] == 3
    assert agg["chi2_n_tested"] == 2                     # the all-fail anchor has no test
    assert agg["chi2_n_p_below_alpha"] == 1
    assert agg["share_explained_success_eta2"]["n_defined"] == 2
    e = [a["stats"]["success"]["eta2"] for a in anchors[:2]]
    np.testing.assert_allclose(agg["share_explained_success_eta2"]["mean"], np.mean(e))
    assert set(agg["bc_success_at_prefix0_per_task"]) == {"1", "2"}
    assert agg["bc_success_at_prefix0_per_task"]["2"] == 0.0
    np.testing.assert_allclose(agg["bc_success_at_prefix0_task_mean"],
                               (anchors[0]["stats"]["success"]["pooled_mean"] + 0.0) / 2)
    assert len(agg["per_anchor"]) == 3


def test_gamma_is_a_parameter_and_defaults_to_the_cube_value():
    success = np.zeros((K, R), int)
    success[:, :30] = 1
    steps = np.full((K, R), 10)
    st = candidate_effect_stats(success, steps)
    assert st["gamma"] == 0.98
    st99 = candidate_effect_stats(success, steps, 0.99)
    assert st99["gamma"] == 0.99
    np.testing.assert_allclose(st99["discounted"]["pooled_mean"], 0.3 * 0.99 ** 10)
    # the success outcome does not depend on the discount
    assert st99["success"] == st["success"] and st99["chi2"] == st["chi2"]


def test_env_table_keeps_cube_and_adds_antmaze():
    c, m = ENVS["cube"], ENVS["antmaze"]
    assert c["env"].format(task=3) == "cube-single-play-singletask-task3-v0"
    assert (c["prefix_lens"], c["gamma"], c["flow"], c["flow_epoch"]) == (
        "0,50,100", 0.98, "cube-single-play", 500000)
    assert c["seed_global_numpy"] is False
    assert m["env"].format(task=3) == "antmaze-medium-navigate-singletask-task3-v0"
    assert (m["prefix_lens"], m["gamma"], m["flow"], m["flow_epoch"]) == (
        "0,250,500", 0.99, "antmaze-medium-navigate", 500000)
    assert m["seed_global_numpy"] is True
    # cube file names are the ones already on disk; antmaze gets its own
    assert default_out_name("cube", 1) == "diag_first_u_effect_bc_task1.json"
    assert default_out_name("cube", 1, smoke=True) == "diag_first_u_effect_bc_task1_smoke.json"
    assert default_out_name("cube") == "diag_first_u_effect_bc_aggregate.json"
    assert default_out_name("antmaze", 4) == "diag_first_u_effect_bc_antmaze_task4.json"
    assert default_out_name("antmaze", 4, smoke=True) == "diag_first_u_effect_bc_antmaze_task4_smoke.json"
    assert default_out_name("antmaze") == "diag_first_u_effect_bc_antmaze_aggregate.json"
