"""Factored-FB induced-reward actor in psmgoal (actor_value = measure_reward_raw | _softmax).

Checks: the raw/softmax value against a hand computation on a tiny mesh; the centered ridge
task fit against numpy; the actor loss gives zero gradient to the basis; one update with the
arm on is finite, moves the actor and writes the index state; the ridge eval coefficient is
on the sqrt(D) sphere; the same-trajectory future-goal sampler stays inside the trajectory.

Run in its own process: JAX_PLATFORMS=cpu .venv/bin/python -m pytest
tests/test_psmgoal_fbtrick.py -q -p no:cacheprovider
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.psmgoal import PSMGoalAgent, get_config, induced_value, kernel_reward, parse_hit_spec, ridge_task
from utils.datasets import future_goal_idxs

OB, DA, Z, CODE, N = 6, 3, 8, 4, 8


def _cfg(**over):
    c = get_config()
    c.z_dim = Z
    c.max_log_seed = CODE
    c.batch_size = N
    c.k_goals = 4
    c.gpi_num_u = 5
    c.num_inference_steps = 2
    c.infer_batch = N
    c.allow_untrained_flow = True
    c.measure.hidden_dim = 16
    c.w.hidden_dim = 16
    c.l.hidden_dim = 16
    c.actor.hidden_dim = 16
    c.flow.hidden_dims = (16, 16)
    c.fb_k_anchor = 4
    c.fb_k_u = 2
    c.fb_hit_spec = "0:3:0.5"
    for k, v in over.items():
        setattr(c, k, v)
    return c


def _agent(**over):
    return PSMGoalAgent.create(0, np.zeros((2, OB), np.float32), np.zeros((2, DA), np.float32),
                               _cfg(**over))


def _batch(n=N, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "observations": rng.standard_normal((n, OB)).astype(np.float32),
        "next_observations": rng.standard_normal((n, OB)).astype(np.float32),
        "noise_preimage": rng.standard_normal((n, DA)).astype(np.float32),
        "fb_goals": rng.standard_normal((n, OB)).astype(np.float32),
        "index": np.arange(n).astype(np.int32),
        "rewards": -np.ones((n,), np.float32),
    }


def test_induced_value_matches_hand():
    M = np.array([[1.0, 2.0, -1.0], [0.5, 0.0, 3.0]], np.float32)
    r = np.array([[0.2, -0.4, 1.0], [1.0, 2.0, -1.0]], np.float32)
    raw = np.array([(1 * .2 + 2 * -.4 + -1 * 1) / 3, (.5 * 1 + 0 * 2 + 3 * -1) / 3])
    np.testing.assert_allclose(induced_value(M, r, "measure_reward_raw", 1.0), raw, rtol=1e-6)
    for temp in (1.0, 0.5):
        e = np.exp(M / temp)
        p = e / e.sum(1, keepdims=True)
        np.testing.assert_allclose(induced_value(M, r, "measure_reward_softmax", temp),
                                   (p * r).sum(1), rtol=1e-5)


@pytest.mark.parametrize("mode", ["measure_reward_raw", "measure_reward_softmax"])
def test_actor_q_matches_hand_on_mesh(mode):
    """The actor loss's Q equals mean/softmax over the pool of M_ij * r_ij computed by hand
    from single-point M calls and r_ij = (f_j - mu) . w_i / sqrt(D)."""
    ag = _agent(train_actor=True, actor_value=mode, actor_temp=0.0)
    b = _batch()
    rng = np.random.default_rng(3)
    obs, pool = jnp.asarray(b["observations"]), jnp.asarray(b["next_observations"])
    w = jnp.asarray(rng.standard_normal((N, Z)), jnp.float32)
    fc = jnp.asarray(rng.standard_normal((N, Z)), jnp.float32)
    noise = jnp.asarray(rng.standard_normal((N, DA)), jnp.float32)
    u_data = jnp.asarray(b["noise_preimage"])
    _, info = ag.actor_loss_fbtrick(ag.actor.params, obs, pool, w, fc, noise, u_data)
    # the actor's latent, recomputed
    from utils.psm_networks import tanh_gaussian_sample
    mu, ls = ag.actor(obs, w)
    u, _ = tanh_gaussian_sample(mu, ls, noise, float(ag.config["u_clip"]))
    Mh = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            Mh[i, j] = float(ag.M(obs[i:i + 1], u[i:i + 1], pool[j:j + 1], w[i])[0])
    rh = np.asarray(w) @ np.asarray(fc).T / np.sqrt(Z)
    if mode == "measure_reward_raw":
        q = (Mh * rh).mean(1)
    else:
        p = np.exp(Mh - Mh.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        q = (p * rh).sum(1)
    np.testing.assert_allclose(float(info["actor_q"]), q.mean(), rtol=1e-4, atol=1e-6)


def test_ridge_task_matches_numpy():
    rng = np.random.default_rng(0)
    M_, D = 50, 6
    f = rng.standard_normal((M_, D)).astype(np.float32)
    mu = f.mean(0) + 0.1
    X = f - f.mean(0)
    cov = (X.T @ X / M_ + 0.05 * np.eye(D)).astype(np.float32)
    r = rng.standard_normal((3, M_)).astype(np.float32)
    got = np.asarray(ridge_task(jnp.asarray(f), jnp.asarray(mu), jnp.asarray(cov), jnp.asarray(r),
                                0.01, 1e-6, 1e-6))
    lam = 0.01 * np.trace(cov) / D
    m_c = (r - r.mean(1, keepdims=True)) @ (f - mu) / M_
    c = np.linalg.solve(cov + lam * np.eye(D), m_c.T).T
    want = np.sqrt(D) * c / np.linalg.norm(c, axis=1, keepdims=True)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(got, axis=1), np.sqrt(D), rtol=1e-5)
    # constant reward -> neutral 1_D
    const = np.asarray(ridge_task(jnp.asarray(f), jnp.asarray(mu), jnp.asarray(cov),
                                  jnp.ones((1, M_)), 0.01, 1e-6, 1e-6))
    np.testing.assert_allclose(const, np.ones((1, D)))


def test_kernel_reward_matches_numpy():
    rng = np.random.default_rng(1)
    g = rng.standard_normal((3, OB)).astype(np.float32)
    p = rng.standard_normal((5, OB)).astype(np.float32)
    got = np.asarray(kernel_reward(jnp.asarray(g), jnp.asarray(p), parse_hit_spec("1:3:0.4")))
    d = ((g[:, None, 1:3] - p[None, :, 1:3]) / 0.4) ** 2
    lg = -0.5 * d.sum(-1)
    e = np.exp(lg - lg.max(1, keepdims=True))
    np.testing.assert_allclose(got, e / e.sum(1, keepdims=True), rtol=1e-5)


def test_actor_loss_gives_basis_zero_gradient():
    ag = _agent(train_actor=True, actor_value="measure_reward_softmax")
    b = _batch()
    rng = np.random.default_rng(2)
    w = jnp.asarray(rng.standard_normal((N, Z)), jnp.float32)
    fc = jnp.asarray(rng.standard_normal((N, Z)), jnp.float32)
    noise = jnp.asarray(rng.standard_normal((N, DA)), jnp.float32)

    def loss_of_basis(bp):
        a = ag.replace(basis=ag.basis.replace(params=bp))
        return a.actor_loss_fbtrick(a.actor.params, jnp.asarray(b["observations"]),
                                    jnp.asarray(b["next_observations"]), w, fc, noise,
                                    jnp.asarray(b["noise_preimage"]))[0]

    g = jax.grad(loss_of_basis)(ag.basis.params)
    assert all(float(jnp.max(jnp.abs(x))) == 0.0 for x in jax.tree_util.tree_leaves(g))
    # while the actor does get a gradient
    ga = jax.grad(lambda ap: ag.actor_loss_fbtrick(
        ap, jnp.asarray(b["observations"]), jnp.asarray(b["next_observations"]), w, fc, noise,
        jnp.asarray(b["noise_preimage"]))[0])(ag.actor.params)
    assert any(float(jnp.max(jnp.abs(x))) > 0.0 for x in jax.tree_util.tree_leaves(ga))


@pytest.mark.parametrize("mode", ["measure_reward_raw", "measure_reward_softmax"])
def test_update_on_is_finite_and_writes_index_state(mode):
    ag = _agent(train_actor=True, actor_value=mode)
    b = _batch()
    new, info = ag.update(b)
    for k in ("actor_q", "actor_loss", "psm_loss", "actor_bc_err", "fb_cov_trace"):
        assert np.isfinite(float(info[k])), k
    assert any(not jnp.array_equal(x, y) for x, y in zip(
        jax.tree_util.tree_leaves(ag.actor.params), jax.tree_util.tree_leaves(new.actor.params)))
    assert float(new.fb_filled) == 1.0
    np.testing.assert_array_equal(np.asarray(new.fb_anchor), b["next_observations"][:4])
    # anchors stay fixed after the first write
    new2, _ = new.update(_batch(seed=1))
    np.testing.assert_array_equal(np.asarray(new2.fb_anchor), b["next_observations"][:4])
    # the index tower moved only by the slow EMA
    d0 = jax.tree_util.tree_leaves(ag.fb_zenc)[0]
    d1 = jax.tree_util.tree_leaves(new.fb_zenc)[0]
    assert np.allclose(d0, d1, atol=1e-3)


def test_ridge_eval_coefficient_on_sphere_and_acts_with_actor():
    ag = _agent(train_actor=True, actor_value="measure_reward_raw", coef_source="ridge",
                acting="distill", eval_redistill=False)
    ag, _ = ag.update(_batch())
    rel = _batch(n=32, seed=5)
    rew = np.zeros((32,), np.float32)
    rew[:6] = 1.0
    ev = ag.infer_eval_goals(rel, rew)
    assert np.isclose(float(jnp.linalg.norm(ev.eval_w_star)), np.sqrt(Z), rtol=1e-4)
    # the actor is untouched by eval (no re-distillation)
    assert all(jnp.array_equal(x, y) for x, y in zip(
        jax.tree_util.tree_leaves(ag.actor.params), jax.tree_util.tree_leaves(ev.actor.params)))
    a = ev.sample_actions(rel["observations"][0], seed=jax.random.PRNGKey(0))
    assert a.shape == (DA,) and np.all(np.isfinite(np.asarray(a)))


def test_future_goal_idxs_stay_in_trajectory():
    terminal_locs = np.array([4, 9, 14])
    idxs = np.repeat(np.arange(15), 50)
    g = future_goal_idxs(idxs, terminal_locs, 15, rng=np.random.default_rng(0))
    ends = terminal_locs[np.searchsorted(terminal_locs, idxs)]
    assert np.all(g >= idxs) and np.all(g <= ends)
    # every future row of row 0's trajectory is reachable
    assert set(g[idxs == 0]) == {0, 1, 2, 3, 4}


def test_defaults_are_off():
    c = get_config()
    assert c.actor_value == "none" and c.train_actor is False and c.coef_source == "lp"
    with pytest.raises(AssertionError):
        _agent(actor_value="measure_reward_raw")          # needs train_actor
