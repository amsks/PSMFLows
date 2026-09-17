"""CPU unit tests for tools/diag_measure_vs_scalar_q.py's pure half.

The GPU half restores two Stage-C checkpoints and the frozen flow, so it is not
exercised here. The metric functions are: shapes on random inputs, Spearman of a
monotone pair = 1, a perfectly Bellman-consistent synthetic psi has zero residual along
w and along every direction, and the row selector oversamples success rows as asked.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.diag_measure_vs_scalar_q import (
    actor_regret,
    direction_residuals,
    dist_bins_pooled,
    measure_bellman,
    near_far_split,
    norm_bin_table,
    pairwise_rank_corr,
    pearson,
    per_state_ranking,
    reduce_pess,
    residual_stats,
    select_rows,
    spearman,
    subset_ranking,
    summarize_rho,
    task_vector_readout,
    top_bin_frac,
    two_way_decomposition,
)


def test_two_way_decomposition_index_blind_and_action_blind():
    rng = np.random.default_rng(8)
    S, Ku, Ki = 6, 16, 12
    # psi that IGNORES the index: Q(s, u, idx) = f(s, u); the index share and the
    # interaction are 0 and the u share is 1
    Q = np.repeat(rng.normal(size=(S, Ku, 1)), Ki, axis=2)
    d = two_way_decomposition(Q)
    assert d['n_states'] == S and d['Ku'] == Ku and d['Kidx'] == Ki
    assert abs(d['frac_u_mean'] - 1.0) < 1e-9 and d['frac_index_mean'] < 1e-9 and d['frac_interaction_mean'] < 1e-9
    assert d['std_index_mean'] < 1e-9
    # every index column ranks the u identically
    assert abs(pairwise_rank_corr(Q, 2)['mean'] - 1.0) < 1e-9
    # psi that IGNORES the action: the u share is 0
    Q = np.repeat(rng.normal(size=(S, 1, Ki)), Ku, axis=1)
    d = two_way_decomposition(Q)
    assert d['frac_u_mean'] < 1e-9 and abs(d['frac_index_mean'] - 1.0) < 1e-9
    assert abs(pairwise_rank_corr(Q, 1)['mean'] - 1.0) < 1e-9
    # additive psi: no interaction; the three shares sum to 1 on any grid
    Q = rng.normal(size=(S, Ku, 1)) + rng.normal(size=(S, 1, Ki))
    d = two_way_decomposition(Q)
    assert d['frac_interaction_mean'] < 1e-9
    Q = rng.normal(size=(S, Ku, Ki))
    d = two_way_decomposition(Q)
    assert abs(d['frac_u_mean'] + d['frac_index_mean'] + d['frac_interaction_mean'] - 1.0) < 1e-9
    r = pairwise_rank_corr(Q, 2)
    assert abs(r['mean']) < 0.2 and r['n_slices'] == Ki


def test_norm_bin_table_and_top_bin_frac():
    S, K = 5, 8
    norms = np.tile(np.arange(K, dtype=float), (S, 1))          # 0..7 in every state
    vals = {'v': norms * 2.0, 'c': (norms >= 6).astype(float)}
    edges, rows = norm_bin_table(norms, vals, n_bins=4)
    assert len(rows) == 4 and sum(r['n'] for r in rows) == S * K
    assert abs(rows[0]['v'] - 1.0) < 1e-12 and abs(rows[-1]['c'] - 1.0) < 1e-12   # bin 0 = norms {0,1}; bin 3 = {6,7}
    assert rows[0]['n'] == 2 * S and rows[-1]['n'] == 2 * S
    idx_top = np.full(S, K - 1)
    idx_low = np.zeros(S, int)
    assert top_bin_frac(norms, idx_top, edges[-2]) == 1.0
    assert top_bin_frac(norms, idx_low, edges[-2]) == 0.0


def test_task_vector_branch_readout_and_consistent_residual():
    """policy_index=task_vector: the value is [mean - kappa unc](psi(s,u,w)^T w) with no
    index panel, and the Bellman bootstrap is psi_bar at the actor's latent. A psi that
    equals phi(s') + gamma * that bootstrap has zero residual along w."""
    rng = np.random.default_rng(6)
    P, N, z, gamma, kappa = 2, 30, 16, 0.98, 0.5
    psi_out = rng.normal(size=(P, N, z))
    w = rng.normal(size=z)
    v = task_vector_readout(psi_out, w, kappa)
    assert v.shape == (N,)
    assert np.allclose(v, np.minimum(psi_out[0] @ w, psi_out[1] @ w))   # kappa=0.5, P=2 is the min
    phi_next = rng.normal(size=(N, z))
    boot = np.broadcast_to(rng.normal(size=(N, z)), (P, N, z)).copy()   # psi_bar(s', pi(s'), w)
    lhs = np.broadcast_to(phi_next + gamma * boot[0], (P, N, z)).copy()
    A = measure_bellman(lhs, boot, phi_next, w, gamma, kappa)
    assert A['lhs_w'].shape == (N,) and residual_stats(A['lhs_w'], A['rhs_w'])['rms'] < 1e-9
    assert direction_residuals(A['lhs_vec'], A['rhs_vec'], w, n_dirs=3)['along_w_rel'] < 1e-9


def test_actor_regret_bounds_and_shapes():
    rng = np.random.default_rng(7)
    S, K = 15, 16
    Vm, Qs = rng.normal(size=(S, K)), rng.normal(size=(S, K))
    # the actor sits exactly at each panel's best latent: zero regret, always in the top-k
    r = actor_regret(Vm, Qs, Vm.max(1), Qs.max(1), topk=4)
    assert set(r) == {'scalar', 'measure'}
    for k in ('scalar', 'measure'):
        assert abs(r[k]['regret_norm_mean']) < 1e-12 and r[k]['frac_in_panel_top4'] == 1.0
        assert r[k]['frac_beats_panel_max'] == 0.0 and 0.0 < r[k]['regret_random_norm_mean'] < 1.0
    # at the worst latent: regret 1, never in the top-k; above the max: negative regret
    r = actor_regret(Vm, Qs, Vm.min(1), Qs.max(1) + 1.0, topk=4)
    assert abs(r['measure']['regret_norm_mean'] - 1.0) < 1e-12 and r['measure']['frac_in_panel_top4'] == 0.0
    assert r['scalar']['frac_beats_panel_max'] == 1.0 and r['scalar']['regret_norm_mean'] < 0.0


def test_spearman_and_pearson_identical_and_reversed():
    x = np.random.default_rng(0).normal(size=50)
    assert abs(spearman(x, x) - 1.0) < 1e-9
    assert abs(spearman(x, 3 * x + 2) - 1.0) < 1e-9
    assert abs(spearman(x, -x) + 1.0) < 1e-9
    assert abs(pearson(x, 3 * x + 2) - 1.0) < 1e-9


def test_reduce_pess_matches_min_for_two_members():
    x = np.random.default_rng(1).normal(size=(2, 7, 3))
    # kappa = 0.5 with P = 2 is the exact min: mean - 0.5 |a - b|
    assert np.allclose(reduce_pess(x, 0.5), x.min(0))
    assert np.allclose(reduce_pess(x, 0.0), x.mean(0))


def test_consistent_synthetic_psi_has_zero_residual():
    rng = np.random.default_rng(2)
    P, N, z, gamma, kappa = 2, 40, 16, 0.98, 0.5
    phi_next = rng.normal(size=(N, z))
    w = rng.normal(size=z)
    # (i) members AGREE: the scalar reduction of psi^T w and the per-component reduction
    # of psi coincide, so lhs = phi(s') + gamma * boot is consistent along w and along
    # every direction.
    boot = np.broadcast_to(rng.normal(size=(N, z)), (P, N, z)).copy()
    lhs = np.broadcast_to(phi_next + gamma * reduce_pess(boot, kappa), (P, N, z)).copy()
    A = measure_bellman(lhs, boot, phi_next, w, gamma, kappa)
    assert A['lhs_w'].shape == (N,) and A['rhs_w'].shape == (N,) and A['lhs_w_p'].shape == (P, N)
    assert residual_stats(A['lhs_w'], A['rhs_w'])['rms'] < 1e-9
    assert residual_stats(A['lhs_w'], A['rhs_w_mean'])['rms'] < 1e-9
    d = direction_residuals(A['lhs_vec'], A['rhs_vec'], w, n_dirs=5)
    assert d['along_w_rel'] < 1e-9 and d['random_dirs_rel_max'] < 1e-9
    assert d['full_vector_rms_norm'] < 1e-9 and len(d['random_dirs_rel']) == 5
    # (ii) members DISAGREE: consistent with the plain-mean target along w, and with the
    # per-component pessimistic target along every direction; the scalar pessimistic
    # target then differs (mean - kappa*|.| is not linear), which is the documented
    # difference between the two along-w readouts.
    boot = rng.normal(size=(P, N, z))
    lhs = np.broadcast_to(phi_next + gamma * boot.mean(0), (P, N, z)).copy()
    A = measure_bellman(lhs, boot, phi_next, w, gamma, kappa)
    assert residual_stats(A['lhs_w'], A['rhs_w_mean'])['rms'] < 1e-9
    assert residual_stats(A['lhs_w'], A['rhs_w'])['rms'] > 0.0
    lhs = np.broadcast_to(phi_next + gamma * reduce_pess(boot, kappa), (P, N, z)).copy()
    A = measure_bellman(lhs, boot, phi_next, w, gamma, kappa)
    d = direction_residuals(A['lhs_vec'], A['rhs_vec'], w, n_dirs=5)
    assert d['along_w_rel'] < 1e-9 and d['random_dirs_rel_max'] < 1e-9


def test_residual_stats_relative_to_std():
    lhs = np.array([0.0, 2.0, 4.0, 6.0])
    rhs = lhs + 1.0
    s = residual_stats(lhs, rhs)
    assert abs(s['mean'] + 1.0) < 1e-12 and abs(s['rms'] - 1.0) < 1e-12
    assert abs(s['rms_over_std_lhs'] - 1.0 / lhs.std()) < 1e-9


def test_per_state_ranking_identical_and_shapes():
    rng = np.random.default_rng(3)
    S, K = 12, 16
    Q = rng.normal(size=(S, K))
    r = per_state_ranking(Q, Q, topk=4)
    assert r['rho'].shape == (S,) and np.allclose(r['rho'], 1.0)
    assert r['hit_topk'].all() and np.allclose(r['regret_norm'], 0.0)
    assert (r['regret_random_norm'] > 0).all() and (r['regret_random_norm'] < 1).all()
    r2 = per_state_ranking(Q, -Q, topk=4)
    assert np.allclose(r2['rho'], -1.0) and not r2['hit_topk'].any()
    assert np.allclose(r2['regret_norm'], 1.0)
    s = summarize_rho(r['rho'])
    assert s['n'] == S and s['frac_gt_0.3'] == 1.0 and s['frac_lt_0'] == 0.0


def test_coverage_helpers_shapes():
    rng = np.random.default_rng(4)
    S, K = 10, 20
    V, Q = rng.normal(size=(S, K)), rng.normal(size=(S, K))
    dist = rng.uniform(size=(S, K))
    near, far = near_far_split(V, V, dist)
    assert near.shape == (S,) and far.shape == (S,) and np.allclose(near, 1.0) and np.allclose(far, 1.0)
    bins = dist_bins_pooled(V, Q, dist, 4)
    assert len(bins) == 4 and sum(b['n'] for b in bins) == S * K
    keep = np.zeros((S, K), bool)
    keep[:, :10] = True
    assert subset_ranking(V, V, keep, min_n=8).shape == (S,)
    assert subset_ranking(V, V, np.zeros((S, K), bool), min_n=8).shape == (0,)


def test_select_rows_oversamples_success():
    rng = np.random.default_rng(5)
    n = 5000
    rewards = -np.ones(n)
    rewards[rng.choice(n, 100, replace=False)] = 0.0       # 2% success
    valid = np.ones(n)
    valid[:7] = 0.0
    rows, is_uniform, is_success = select_rows(rewards, valid, n_rows=1000, n_success_min=60, seed=9)
    assert is_uniform.sum() == 1000 and len(rows) == len(set(rows.tolist()))
    assert is_success.sum() >= 60 and (rewards[rows[is_success]] == 0.0).all()
    assert not set(rows.tolist()) & set(range(7))           # invalid rows never selected
    # deterministic in the seed
    rows2, _, _ = select_rows(rewards, valid, n_rows=1000, n_success_min=60, seed=9)
    assert np.array_equal(rows, rows2)
