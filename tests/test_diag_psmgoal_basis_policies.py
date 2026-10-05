"""CPU tests for tools/diag_psmgoal_basis_policies.py on a tiny synthetic basis."""
import numpy as np

import utils.xla_guard  # noqa: F401
import jax
import jax.numpy as jnp

from tools.diag_psmgoal_basis_policies import (add_stats, code_bits, empty_stats, finalize, fit_w_td,
                                                mesh_sums, normalized_td_error, participation_ratio,
                                                policy_latents, td_fixed_point, two_way_decomposition,
                                                within_group_fraction)
from utils.psm_proto import proto_latents, proto_seed_ints


def test_two_way_decomposition_additive():
    rng = np.random.default_rng(0)
    a, c = rng.normal(size=(8, 1)) * 3.0, rng.normal(size=(1, 50))
    d = two_way_decomposition(5.0 + a + c)
    assert abs(d["frac_interaction"]) < 1e-10
    assert abs(d["frac_policy"] + d["frac_cell"] - 1.0) < 1e-10
    ss_a, ss_c = (a - a.mean()) ** 2 * 50, (c - c.mean()) ** 2 * 8
    assert np.isclose(d["frac_policy"], ss_a.sum() / (ss_a.sum() + ss_c.sum()))
    d2 = two_way_decomposition(rng.normal(size=(8, 50)) + 0 * a)
    assert np.isclose(d2["frac_policy"] + d2["frac_cell"] + d2["frac_interaction"], 1.0)


def test_policy_independent_M_has_zero_policy_fraction():
    rng = np.random.default_rng(1)
    phi, b = rng.normal(size=(40, 4)), rng.normal(size=40)
    w = np.tile(rng.normal(size=(1, 4)), (6, 1))
    d = two_way_decomposition(w @ phi.T + b[None])
    assert d["frac_policy"] < 1e-12 and d["frac_cell"] > 1 - 1e-12


def test_participation_ratio_and_within_fraction():
    X = np.zeros((10, 5))
    X[:, 0] = np.arange(10)
    assert np.isclose(participation_ratio(X, center=False)["pr"], 1.0)
    assert np.isclose(participation_ratio(np.eye(5), center=False)["pr"], 5.0)
    V = np.repeat(np.arange(6.0)[:, None], 4, 1)          # constant within a cell
    assert within_group_fraction(V)["frac_within"] < 1e-12


def test_new_seed_policy_construction():
    width, d_a, rows = 16, 5, np.arange(100, 164)
    bits = code_bits(np.array([12345]), width)[0]
    assert bits.shape == (width,) and set(np.unique(bits)) <= {0.0, 1.0}
    assert int((bits * 2 ** np.arange(width)).sum()) == 12345     # LSB-first
    u_train = np.asarray(policy_latents(bits, rows, width, d_a, 3.0, 0))
    u_new = np.asarray(policy_latents(bits, rows, width, d_a, 3.0, 12345))
    z = jnp.broadcast_to(jnp.asarray(bits), (rows.size, width))
    ref = proto_latents(proto_seed_ints(z, rows, width), d_a, 3.0, jax.random.PRNGKey(0))
    np.testing.assert_allclose(u_train, np.asarray(ref))            # base 0 == training policy
    assert u_new.shape == (rows.size, d_a) and np.abs(u_new).max() <= 3.0
    assert np.abs(u_new - u_train).mean() > 0.5                   # a different policy
    np.testing.assert_allclose(u_new, np.asarray(policy_latents(bits, rows, width, d_a, 3.0, 12345)))


def _synthetic_stats(seed, D=4, N=24, gamma=0.9):
    """A tiny linear basis: P = [phi, b] on the mesh, Q = the same map at a shifted input."""
    rng = np.random.default_rng(seed)
    Wphi = rng.normal(size=(3, D + 1))
    x = rng.normal(size=(N, N, 3))
    P = np.tanh(x @ Wphi)
    Q = np.tanh((0.7 * x + 0.3 * rng.normal(size=x.shape)) @ Wphi)
    acc = add_stats(empty_stats(D + 1), jax.device_get(mesh_sums(jnp.asarray(P), jnp.asarray(Q))))
    return finalize(acc), P, Q


def test_stats_match_direct_nmse():
    gamma = 0.9
    st, P, Q = _synthetic_stats(0, gamma=gamma)
    w = np.array([0.3, -1.0, 0.5, 2.0])
    W = np.append(w, 1.0)
    N = P.shape[0]
    off = ~np.eye(N, dtype=bool)
    M, Mb = P @ W, Q @ W
    direct = np.mean((M - gamma * Mb)[off] ** 2) / np.var(gamma * Mb[off])
    assert np.isclose(normalized_td_error(st, w, gamma)["nmse"], direct, rtol=1e-4)


def test_adam_td_reaches_fixed_point():
    gamma = 0.9
    st, _, _ = _synthetic_stats(1, gamma=gamma)
    w_fp = td_fixed_point(st, gamma)
    w = fit_w_td(st, np.zeros(4), gamma, steps=20000, lr=1e-2, tau=0.05, project=False)
    np.testing.assert_allclose(w, w_fp, atol=2e-2 * max(1.0, np.abs(w_fp).max()))
    ws = fit_w_td(st, np.ones(4), gamma, steps=200, lr=1e-2, tau=0.05, project=True)
    assert np.isclose(np.linalg.norm(ws), 2.0, atol=1e-4)


def test_spread_ratio_and_argmax_disagreement():
    from tools.diag_psmgoal_basis_policies import argmax_disagreement, spread_ratio
    s = np.arange(5.0)[:, None, None]
    V = np.broadcast_to(s, (5, 3, 4)).copy()                  # varies only across states
    assert spread_ratio(V)["ratio_u_over_s"] < 1e-12
    Q = np.zeros((2, 3, 4))
    Q[0, :, 1] = 1.0
    Q[1, :, 1] = 1.0
    Q[1, 2, 3] = 5.0                                           # state 2 argmax moves
    d = argmax_disagreement(Q)
    assert d[0, 0] == 0 and np.isclose(d[0, 1], 1 / 3) and np.isclose(d[1, 0], 1 / 3)
