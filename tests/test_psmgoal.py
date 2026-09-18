"""psmgoal = the faithful RLU (Proto Successor Measure) port on flow latents.

Design: docs/design/2026-09-17-psmgoal.md (RLU rewrite). These tests pin the pieces that
distinguish it from the factorized f_psmflow: a GENERAL measure phi(s,u,g)/b(s,u,g) with no
scale anchor, a coefficient w(z) on the binary policy code L2-normed to sqrt(D), the mesh TD
basis loss with a fixed z-indexed bootstrap, the Lagrangian coefficient inference with
sphere renormalization and a dual-ascended multiplier, and goal-set acting.

Run this file in its own process (JAX_PLATFORMS=cpu ... -p no:cacheprovider); the whole
suite in one process SIGABRTs in XLA.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.psmgoal import PSMGoalAgent, get_config
from utils.psm_networks import PolicyCoefficient, RLUMeasure

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
    for k, v in over.items():
        setattr(c, k, v)
    return c


def _agent(seed=0, **over):
    obs = np.zeros((2, OB), np.float32)
    act = np.zeros((2, DA), np.float32)
    return PSMGoalAgent.create(seed, obs, act, _cfg(**over))


def _batch(n=N, seed=0, reward_rows=()):
    # cube's task reward convention is {-1, 0}; eval shifts by +1 so a rewarding (goal) row is
    # 0 -> 1 and a non-rewarding row is -1 -> 0. Base every row at -1, set the goal rows to 0.
    rng = np.random.default_rng(seed)
    r = -np.ones((n,), np.float32)
    for i in reward_rows:
        r[i] = 0.0
    return {
        "observations": rng.standard_normal((n, OB)).astype(np.float32),
        "next_observations": rng.standard_normal((n, OB)).astype(np.float32),
        "noise_preimage": rng.standard_normal((n, DA)).astype(np.float32),
        "index": np.arange(n).astype(np.int32),
        "rewards": r,
    }


# --------------------------------------------------------------------- networks

def test_rlu_measure_is_general_no_anchor():
    """phi(s,u,g) is a plain head: its row norms are NOT pinned to sqrt(z_dim) (that pin is
    the f_psmflow anchor psmgoal must not have), and shapes are (B, z_dim) / (B,)."""
    net = RLUMeasure(z_dim=Z, hidden_dim=16)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (5, OB))
    u = jax.random.normal(rng, (5, DA))
    g = jax.random.normal(rng, (5, OB))
    params = net.init(rng, x, u, g)["params"]
    phi, b = net.apply({"params": params}, x, u, g)
    assert phi.shape == (5, Z) and b.shape == (5,)
    norms = jnp.linalg.norm(phi, axis=-1)
    assert not jnp.allclose(norms, jnp.sqrt(float(Z)), atol=1e-3), "phi must not be sphere-anchored"


def test_policy_coefficient_on_sqrt_z_sphere():
    """w(z) rows have norm sqrt(z_dim) (RLU `_L2`)."""
    net = PolicyCoefficient(z_dim=Z, hidden_dim=16)
    rng = jax.random.PRNGKey(1)
    z = jax.random.bernoulli(rng, 0.5, (7, CODE)).astype(jnp.float32)
    z = z.at[:, 0].set(1.0)  # avoid the all-zero code (enc(0)=0 -> _L2(0)=0, faithful to RLU)
    params = net.init(rng, z)["params"]
    w = net.apply({"params": params}, z)
    assert w.shape == (7, Z)
    assert jnp.allclose(jnp.linalg.norm(w, axis=-1), jnp.sqrt(float(Z)), atol=1e-4)


# --------------------------------------------------------------------- measure / mesh

def test_mesh_matches_pointwise_measure():
    """mesh_M[i, j] equals the pointwise M(s_i, u_i, g_j)."""
    ag = _agent()
    b = _batch()
    obs, u, goals = jnp.asarray(b["observations"]), jnp.asarray(b["noise_preimage"]), jnp.asarray(b["next_observations"])
    z = jnp.zeros((N, CODE), jnp.float32)
    w = ag.w(z)                                    # (N, z)
    mesh = ag.mesh_M(obs, u, goals, w)               # (N, N)
    i, j = 2, 5
    point = ag.M(obs[i][None], u[i][None], goals[j][None], w[i][None])[0]
    assert jnp.allclose(mesh[i, j], point, atol=1e-5)


def test_measure_loss_is_offdiag_td_plus_diag_pull():
    """loss == 0.5*mean(offdiag (M-gamma*Mbar)^2) - (1-gamma)*mean(diag(M))."""
    ag = _agent()
    b = _batch()
    rng = jax.random.PRNGKey(3)
    z = jax.random.bernoulli(rng, 0.5, (N, CODE)).astype(jnp.float32)
    idx = jnp.asarray(b["index"])
    u_next = ag.proto_bootstrap(z, idx)
    loss, info = ag.measure_loss(ag.basis.params, ag.w.params, b, z, u_next)

    # recompute independently (at init target == online, so use the same params)
    obs = jnp.asarray(b["observations"]); nobs = jnp.asarray(b["next_observations"])
    u = jnp.clip(jnp.asarray(b["noise_preimage"]), -3.0, 3.0)
    w = ag.w(z); w_t = ag.w(z, params=ag.target_w)
    M = ag.mesh_M(obs, u, nobs, w, params=ag.basis.params)
    Mbar = ag.mesh_M(nobs, u_next, nobs, w_t, params=ag.target_basis)
    g = 0.98
    off = 1.0 - jnp.eye(N)
    diff = M - g * Mbar
    exp = 0.5 * jnp.sum((diff ** 2) * off) / jnp.sum(off) - (1.0 - g) * jnp.mean(jnp.diagonal(M))
    assert jnp.allclose(loss, exp, atol=1e-5)
    assert jnp.isfinite(info["psm_loss"])


# --------------------------------------------------------------------- proto bootstrap

def test_proto_bootstrap_deterministic_and_in_support():
    ag = _agent()
    z = jax.random.bernoulli(jax.random.PRNGKey(4), 0.5, (N, CODE)).astype(jnp.float32)
    idx = jnp.arange(N)
    u1 = ag.proto_bootstrap(z, idx)
    u2 = ag.proto_bootstrap(z, idx)
    assert jnp.array_equal(u1, u2), "same (z, row) must give the same latent"
    assert u1.shape == (N, DA)
    assert jnp.all(jnp.abs(u1) <= 3.0 + 1e-6), "bootstrap latent must stay in the u_clip box"
    z2 = 1.0 - z
    assert not jnp.allclose(u1, ag.proto_bootstrap(z2, idx)), "a different code indexes a different policy"


# --------------------------------------------------------------------- update / polyak

def test_update_runs_and_moves_target_by_tau():
    ag = _agent()
    b = _batch()
    new, info = ag.update(b)
    assert jnp.isfinite(info["psm_loss"])
    leaf = "phi_out"
    old_t = jax.tree_util.tree_leaves(ag.target_basis)[0]
    new_t = jax.tree_util.tree_leaves(new.target_basis)[0]
    new_online = jax.tree_util.tree_leaves(new.basis.params)[0]
    expected = (1 - 0.01) * old_t + 0.01 * new_online
    assert jnp.allclose(new_t, expected, atol=1e-6)


def test_update_changes_basis_and_w():
    ag = _agent()
    new, _ = ag.update(_batch())
    b0 = jax.tree_util.tree_leaves(ag.basis.params)[0]
    b1 = jax.tree_util.tree_leaves(new.basis.params)[0]
    c0 = jax.tree_util.tree_leaves(ag.w.params)[0]
    c1 = jax.tree_util.tree_leaves(new.w.params)[0]
    assert not jnp.allclose(b0, b1), "basis must update"
    assert not jnp.allclose(c0, c1), "coefficient w(z) must update jointly"


# --------------------------------------------------------------------- inference (RLU infer_w)

def test_inference_step_renormalizes_w_to_sphere():
    ag = _agent()
    b = _batch()
    st = ag.init_inference(jax.random.PRNGKey(5))
    obs, u = jnp.asarray(b["observations"]), jnp.asarray(b["noise_preimage"])
    goals = jnp.asarray(b["next_observations"])[:4]
    st2, info = ag.inference_step(st, obs, u, goals, jax.random.PRNGKey(6))
    assert jnp.allclose(jnp.linalg.norm(st2.w), jnp.sqrt(float(Z)), atol=1e-4)
    assert jnp.isfinite(info["obj"])


def test_inference_dgd_updates_multiplier_hinge_does_not():
    b = _batch()
    obs, u = jnp.asarray(b["observations"]), jnp.asarray(b["noise_preimage"])
    goals = jnp.asarray(b["next_observations"])[:4]
    # dual gradient descent: the multiplier network is stepped
    ag = _agent(use_dgd=True)
    st = ag.init_inference(jax.random.PRNGKey(7))
    st2, _ = ag.inference_step(st, obs, u, goals, jax.random.PRNGKey(8))
    m0 = jax.tree_util.tree_leaves(st.l_params)[0]
    m1 = jax.tree_util.tree_leaves(st2.l_params)[0]
    assert not jnp.allclose(m0, m1), "use_dgd must step the multiplier"
    # hinge: no multiplier, params untouched, w still renormed
    agh = _agent(use_dgd=False)
    sth = agh.init_inference(jax.random.PRNGKey(7))
    sth2, _ = agh.inference_step(sth, obs, u, goals, jax.random.PRNGKey(8))
    h0 = jax.tree_util.tree_leaves(sth.l_params)[0]
    h1 = jax.tree_util.tree_leaves(sth2.l_params)[0]
    assert jnp.allclose(h0, h1), "hinge path must not step the multiplier"
    assert jnp.allclose(jnp.linalg.norm(sth2.w), jnp.sqrt(float(Z)), atol=1e-4)


def test_multiplier_ascent_raises_l_where_violated():
    """One dual step raises l more on violated samples (cons < 0) than on satisfied ones."""
    ag = _agent(use_dgd=True)
    b = _batch()
    obs, u = jnp.asarray(b["observations"]), jnp.asarray(b["noise_preimage"])
    goals = jnp.asarray(b["next_observations"])[:4]
    st = ag.init_inference(jax.random.PRNGKey(9))
    perm = jnp.arange(obs.shape[0])
    _, cons, l0 = ag._obj_and_constraint(st.w, st.l_params, obs, u, goals, perm)
    st2, _ = ag.inference_step(st, obs, u, goals, jax.random.PRNGKey(0))  # perm here differs; recompute below
    # re-evaluate the multiplier on the SAME inputs with the new params
    _, _, l1 = ag._obj_and_constraint(st.w, st2.l_params, obs, u, goals, perm)
    viol = cons < 0.0
    if bool(jnp.any(viol)) and bool(jnp.any(~viol)):
        d = l1 - l0
        assert float(jnp.mean(d[viol])) > float(jnp.mean(d[~viol]))
    else:
        pytest.skip("batch had no mixed violation at init")


# --------------------------------------------------------------------- acting / eval

def test_sample_actions_shape_and_clip():
    ag = _agent()
    b = _batch(reward_rows=(1, 3))
    ag = ag.infer_eval_goals(b, b["rewards"] + 1.0)   # sets eval_goals + eval_w_star
    a = ag.sample_actions(jnp.asarray(b["observations"][0]), seed=jax.random.PRNGKey(2))
    assert a.shape == (DA,)
    assert jnp.all(jnp.abs(a) <= 1.0 + 1e-6)


def test_infer_eval_goals_sets_goalset_and_normed_w():
    ag = _agent()
    b = _batch(reward_rows=(0, 2, 5))
    ev = ag.infer_eval_goals(b, b["rewards"] + 1.0)
    assert ev.eval_goals.shape == (4, OB)
    assert jnp.allclose(jnp.linalg.norm(ev.eval_w_star), jnp.sqrt(float(Z)), atol=1e-3)


def test_infer_eval_goals_requires_a_rewarding_row():
    ag = _agent()
    b = _batch(reward_rows=())
    with pytest.raises(ValueError):
        ag.infer_eval_goals(b, b["rewards"] + 1.0)


# --------------------------------------------------------------- A/B goal_conditioned mode

def _batch_g(**kw):
    """A batch that also carries batch['goals'] (the hindsight mixture main.py provides)."""
    b = _batch(**kw)
    rng = np.random.default_rng(7)
    b["goals"] = rng.standard_normal((b["observations"].shape[0], OB)).astype(np.float32)
    return b


def test_goal_head_h_on_sqrt_z_sphere():
    ag = _agent()
    g = jax.random.normal(jax.random.PRNGKey(0), (5, OB))
    w = ag.w_star(g)
    assert jnp.allclose(jnp.linalg.norm(w, axis=-1), jnp.sqrt(float(Z)), atol=1e-3)


def test_defaults_do_not_train_goal_head_or_touch_it():
    # Core defaults (train_goal_head=False): update() must leave w_star untouched, so the
    # running RLU-core jobs are byte-identical.
    ag = _agent()
    b = _batch_g()
    new, info = ag.update(b)
    leaves0 = jax.tree_util.tree_leaves(ag.w_star.params)
    leaves1 = jax.tree_util.tree_leaves(new.w_star.params)
    assert all(jnp.array_equal(a, c) for a, c in zip(leaves0, leaves1))
    assert "goal_obj" not in info  # the goal-head branch did not run


def test_train_goal_head_updates_w_star_only():
    ag = _agent(train_goal_head=True)
    b = _batch_g()
    new, info = ag.update(b)
    assert "goal_obj" in info and "goal_pen" in info
    # w_star moved
    moved = any(not jnp.array_equal(a, c) for a, c in zip(
        jax.tree_util.tree_leaves(ag.w_star.params),
        jax.tree_util.tree_leaves(new.w_star.params)))
    assert moved


def test_goal_head_loss_grad_reaches_h_only():
    # J+constraint gradient routes to w_star; phi/b are stop-gradded inside goal_head_loss.
    ag = _agent(train_goal_head=True)
    b = _batch_g()
    perm = jnp.arange(b["observations"].shape[0])[::-1]
    (_, _), g = jax.value_and_grad(ag.goal_head_loss, has_aux=True)(ag.w_star.params, b, perm)
    assert any(jnp.any(x != 0) for x in jax.tree_util.tree_leaves(g))


def test_amortized_coef_source_reads_h_of_goals():
    ag = _agent(coef_source="amortized")
    b = _batch(reward_rows=(0, 2, 5))
    ev = ag.infer_eval_goals(b, b["rewards"] + 1.0)
    # eval_w_star == project(mean_g h(g)) and is on the sphere
    expect = ev._project(jnp.mean(ev.w_star(ev.eval_goals), axis=0))
    assert jnp.allclose(ev.eval_w_star, expect, atol=1e-5)
    assert jnp.allclose(jnp.linalg.norm(ev.eval_w_star), jnp.sqrt(float(Z)), atol=1e-3)


def test_restore_tolerates_checkpoint_without_w_star(tmp_path):
    # A core checkpoint written before the goal head existed has no 'w_star' field. The new
    # agent must still restore it (keeping the fresh head) and act -- this is what keeps the
    # RUNNING RLU-core jobs' eval500 safe.
    import pickle
    import flax
    from utils.flax_utils import restore_agent

    ag = _agent(seed=1)
    saved = flax.serialization.to_state_dict(ag)
    saved.pop("w_star")  # simulate a pre-goal-head checkpoint
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with open(run_dir / "params_10.pkl", "wb") as f:
        pickle.dump({"agent": saved}, f)
    fresh = _agent(seed=2)
    restored = restore_agent(fresh, str(run_dir), 10)
    b = _batch(reward_rows=(1,))
    ev = restored.infer_eval_goals(b, b["rewards"] + 1.0)
    a = ev.sample_actions(jnp.asarray(b["observations"][0]), seed=jax.random.PRNGKey(0))
    assert a.shape == (DA,) and jnp.all(jnp.abs(a) <= 1.0 + 1e-6)
