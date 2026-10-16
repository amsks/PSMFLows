"""CPU tests for the pure numpy helpers of tools/diag_actor_shift_direction.py on synthetic
samples: the direction statistics, the goal/state variance split, the action-space statistics
and the random-cosine reference."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.diag_actor_shift_direction import (
    action_shift_stats,
    cosine,
    goal_variance_split,
    random_cosine_reference,
    shift_direction_stats,
)


def test_cosine_basic():
    assert abs(cosine([1, 0], [1, 0]) - 1.0) < 1e-12
    assert abs(cosine([1, 0], [0, 1])) < 1e-12
    assert abs(cosine([1, 0], [-2, 0]) + 1.0) < 1e-12
    assert cosine([0, 0], [1, 0]) == 0.0


def test_random_cosine_reference_matches_dimension():
    rng = np.random.default_rng(0)
    r5 = random_cosine_reference(5, rng)
    r50 = random_cosine_reference(50, rng)
    # E|cos| in d dims ~ sqrt(2 / (pi d)) for large d; falls with d
    assert 0.3 < r5 < 0.45
    assert 0.08 < r50 < 0.14
    assert r50 < r5


def test_shift_direction_stats_constant_shift_toward_top_centroid():
    rng = np.random.default_rng(1)
    N, d = 400, 5
    eps = rng.standard_normal((N, d))
    u_bc = np.clip(eps, -3, 3)
    c_top = np.array([2.0, 0, 0, 0, 0])
    u_data = np.array([0, 2.0, 0, 0, 0])
    delta = np.tile(np.array([0.1, 0, 0, 0, 0]), (N, 1))          # a constant step along c_top
    grad = np.tile(np.array([1.0, 0, 0, 0, 0]), (64, 1))
    s = shift_direction_stats(delta, eps, u_bc, c_top, u_data, grad)
    assert abs(s['cos_dbar_ctop'] - 1.0) < 1e-9
    assert abs(s['cos_dbar_udata']) < 1e-9
    assert abs(s['cos_dbar_gbar'] - 1.0) < 1e-9
    assert abs(s['proj_dbar_on_ctop'] - 1.0) < 1e-9
    assert abs(s['mean_explained'] - 1.0) < 1e-9
    assert abs(s['dbar_norm'] - 0.1) < 1e-9
    assert abs(s['cos_ctop_udata']) < 1e-9
    # per draw: the step is +x, so it points toward c_top from any draw with x < 2 (most)
    assert s['frac_pos_to_ctop'] > 0.9
    assert abs(s['cos_delta_grad'] - 1.0) < 1e-9 and s['frac_pos_grad'] == 1.0
    # outward: cos(+x step, draw) > 0 iff the draw's x > 0: half the draws
    assert 0.4 < s['frac_pos_outward'] < 0.6


def test_shift_direction_stats_random_shift_has_no_direction():
    rng = np.random.default_rng(2)
    N, d = 2000, 5
    eps = rng.standard_normal((N, d))
    delta = 0.1 * rng.standard_normal((N, d))                      # zero-mean, no preferred direction
    c_top = rng.standard_normal(d)
    u_data = rng.standard_normal(d)
    grad = rng.standard_normal((64, d))
    s = shift_direction_stats(delta, eps, np.clip(eps, -3, 3), c_top, u_data, grad)
    assert s['mean_explained'] < 0.05
    assert abs(s['cos_delta_to_ctop']) < 0.05
    assert abs(s['frac_pos_to_ctop'] - 0.5) < 0.05
    assert abs(s['cos_delta_outward']) < 0.05


def test_goal_variance_split_all_state_vs_all_goal():
    rng = np.random.default_rng(3)
    S, K, d = 64, 8, 5
    per_state = rng.standard_normal((S, 1, d))
    all_state = np.repeat(per_state, K, axis=1)                    # the same shift for every goal
    g = goal_variance_split(all_state)
    assert g['goal_share'] < 1e-9
    assert abs(g['cos_hind_vs_random_goal'] - 1.0) < 1e-9
    assert g['frac_pos_hind_vs_random_goal'] == 1.0
    all_goal = rng.standard_normal((S, K, d))                      # independent per (state, goal)
    g = goal_variance_split(all_goal)
    assert g['goal_share'] > 0.8
    assert abs(g['cos_hind_vs_random_goal']) < 0.1
    assert abs(g['var_total'] - (g['var_across_goals_fixed_state'] + g['var_across_states'])) < 1e-9


def test_action_shift_stats_toward_data_and_gripper_flip():
    rng = np.random.default_rng(4)
    N, A = 300, 5
    a_bc = np.clip(rng.standard_normal((N, A)) * 0.5, -1, 1)
    a_data = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    a_top = np.array([0.5, 0.0, 0.0, 0.0, 1.0])
    # the actor moves each decoded action 10% of the way toward a_data
    a_act = a_bc + 0.1 * (a_data[None, :] - a_bc)
    s = action_shift_stats(a_bc, a_act, a_data, a_top)
    assert abs(s['cos_action_to_data'] - 1.0) < 1e-9 and s['frac_pos_action_to_data'] == 1.0
    assert s['cos_action_to_top'] > 0.5
    assert abs(s['action_shift_norm'] - 0.1 * s['action_bc_to_data_norm']) < 1e-9
    # moving 10% toward +1 flips exactly the draws with -1/9 < a_bc[4] < 0, and every flip is toward data
    expected_flip = float(np.mean((a_bc[:, 4] < 0) & (0.9 * a_bc[:, 4] + 0.1 > 0)))
    assert abs(s['gripper_flip_rate'] - expected_flip) < 1e-9
    assert abs(s['gripper_flip_toward_data_rate'] - expected_flip) < 1e-9
    assert {k for k in s if k.startswith('shift_mean_')} == {
        'shift_mean_pos_x', 'shift_mean_pos_y', 'shift_mean_pos_z', 'shift_mean_ori', 'shift_mean_gripper'}
    # a full flip of the gripper toward the data sign
    a_act2 = a_bc.copy()
    a_act2[:, 4] = 1.0
    s2 = action_shift_stats(a_bc, a_act2, a_data, a_top)
    neg = float(np.mean(a_bc[:, 4] < 0))
    assert abs(s2['gripper_flip_rate'] - neg) < 1e-9
    assert abs(s2['gripper_flip_toward_data_rate'] - neg) < 1e-9
    assert s2['gripper_actor_agrees_data'] == 1.0
    assert abs(s2['gripper_bc_agrees_data'] - (1.0 - neg)) < 1e-9
