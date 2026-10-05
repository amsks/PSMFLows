"""Pure-numpy helpers of tools/diag_fourrooms_induced_reward.py (no jax, no agent)."""
import numpy as np

from tools import diag_fourrooms_induced_reward as F


def test_layout_and_moves():
    free, cells, idx, T = F.build_grid()
    assert free.sum() == 104 and len(cells) == 104
    assert all(free[r, c] for r, c in F.DOORS) and all(free[r, c] for r, c in F.GOAL_RC)
    # every move is reversible or a bump: the random walk is symmetric
    for s in range(104):
        for a in range(F.A):
            s2 = T[s, a]
            assert s2 == s or s in T[s2]


def test_exact_measure_rows_sum_to_one():
    _, _, _, T = F.build_grid()
    M = F.exact_measure(T, np.full((104, F.A), 0.25), 0.98)
    assert M.shape == (104, F.A, 104)
    np.testing.assert_allclose(M.sum(-1), 1.0, atol=1e-9)
    assert (M >= -1e-12).all()


def test_true_reward_q_iteration_is_shortest_path():
    _, _, idx, T = F.build_grid()
    g = int(idx[F.GOAL_RC[6]])
    dist = F.bfs_dist(T, g)
    assert np.isfinite(dist).all() and dist[g] == 0 and dist.max() > F.FAR
    r = (T == g).astype(np.float64) - 1.0
    Q = F.q_iteration(T, r, (T != g).astype(np.float64), 0.98)
    greedy = Q.argmax(1)
    assert F.shortest_share(T, dist, greedy, g) == 1.0
    starts = np.setdiff1d(np.arange(104), [g])
    done, steps, min_dist = F.rollout(T, lambda c, p: greedy[c], starts, np.zeros(len(starts)), g, dist=dist)
    assert done.all() and (min_dist == 0).all()
    assert F.rollout_summary(done, steps, min_dist)['reach_within1'] == 1.0
    np.testing.assert_array_equal(steps, dist[starts])
    path = F.shortest_path_cells(T, dist, int(idx[F.FIG_START_RC]), g)
    assert path[-1] == g and len(path) == dist[path[0]] + 1


def test_greedy_on_exact_measure_reaches_goal():
    _, _, idx, T = F.build_grid()
    M = F.exact_measure(T, np.full((104, F.A), 0.25), 0.98)
    for rc in F.GOAL_RC:
        g = int(idx[rc])
        greedy = M[:, :, g].argmax(1)
        starts = np.setdiff1d(np.arange(104), [g])
        done, _, _ = F.rollout(T, lambda c, p: greedy[c], starts, np.zeros(len(starts)), g)
        assert done.all()


def test_row_tables_and_gap():
    _, _, idx, T = F.build_grid()
    S = 104
    st = np.repeat(np.arange(S), F.A)
    ac = np.tile(np.arange(F.A), S)
    vals = np.arange(S * F.A, dtype=np.float64)
    np.testing.assert_allclose(F.rows_to_table(st, ac, vals, S), vals.reshape(S, F.A))
    nx = T[st, ac]
    np.testing.assert_allclose(F.cell_mean(nx, np.ones(len(nx)), S), 1.0)
    g = int(idx[F.GOAL_RC[0]])
    dist = F.bfs_dist(T, g)
    score = np.zeros((S, F.A))
    score[:, 0] = 1.0 + np.arange(S)          # best action leads the second by 1 + cell index
    out = F.gap_over_spread(score, dist, g)
    assert out['near'] > 0 and out['far'] > 0 and out['n_far_cells'] > 0
    assert F.argmax_agree(score.argmax(1), np.zeros(S, int), g) == 1.0


def test_reward_variants_scale_and_masks():
    rng = np.random.default_rng(0)
    done = (rng.random(4000) < 0.05).astype(np.float64)
    r_true = done - 1.0
    r_hat = 3.0 * done + rng.normal(0, 0.5, 4000) + 7.0
    variants, fits = F.reward_variants({'x': r_hat}, r_true, done)
    assert set(variants) == {'true', 'const_mask', 'x_mask', 'x_nomask'}
    r, m = variants['x_mask']
    np.testing.assert_allclose(m, 1.0 - done)
    assert abs(r.mean() - r_true.mean()) < 1e-4            # affine fit matches the mean of the real reward
    z, m2 = variants['x_nomask']
    assert (m2 == 1).all() and abs(z.mean()) < 1e-9 and abs(z.std() - 1) < 1e-6
    assert fits['x']['corr'] > 0.5 and fits['x']['affine_scale'] > 0
    # a negatively correlated readout is flipped by the affine fit's sign
    variants2, fits2 = F.reward_variants({'x': -r_hat}, r_true, done)
    assert fits2['x']['affine_scale'] < 0
    np.testing.assert_allclose(variants2['x_nomask'][0], z, atol=1e-9)
