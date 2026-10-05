"""In-support policy extractors for psmgoal (added 2026-09-21).

The measure M(s,u,g)=phi(s,u,g)^T w + b overvalues out-of-support latents, so an actor that
freely maximizes it collapses. These pin two in-support extractors gated by the new
`actor_objective` config key (train_actor=true only):

- awr: advantage-weighted regression. The actor is a behaviour-clone of the DATA latent
  u=noise_preimage, weighted by exp((Q_data - V)/beta) where Q_data reads the measure at the
  data latent and V is a Monte-Carlo prior-draw baseline. The loss target is always u_data, so
  it never queries an out-of-support u.
- bc: plain (unweighted) behaviour clone of u_data, paired with a new acting mode `sfbc` that
  reranks candidate latents drawn from the trained actor by the goal-averaged measure.

Default actor_objective=distill is the existing DSRL path and must stay unchanged.

Run in its own process:
  JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_psmgoal_extractors.py -q -p no:cacheprovider
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from agents.psmgoal import PSMGoalAgent, get_config, tanh_gaussian_logprob
from utils.psm_networks import tanh_gaussian_sample

OB, DA, Z, CODE, N = 6, 3, 8, 4, 8


def _cfg(**over):
    c = get_config()
    c.z_dim = Z
    c.max_log_seed = CODE
    c.batch_size = N
    c.k_goals = 4
    c.gpi_num_u = 5
    c.sfbc_num_u = 5
    c.awr_baseline_k = 6
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


def _changed(t0, t1):
    return any(not jnp.array_equal(a, b) for a, b in zip(
        jax.tree_util.tree_leaves(t0), jax.tree_util.tree_leaves(t1)))


def _same(t0, t1):
    return all(jnp.array_equal(a, b) for a, b in zip(
        jax.tree_util.tree_leaves(t0), jax.tree_util.tree_leaves(t1)))


# ---- fixtures shared by the awr tests: the exact inputs apply_update feeds the actor loss

def _actor_inputs(ag, b):
    c = ag.config
    obs = jnp.asarray(b["observations"])
    goals = jnp.asarray(b["next_observations"])
    z = jnp.zeros((obs.shape[0], CODE), jnp.float32).at[:, 0].set(1.0)
    w_rows = jax.lax.stop_gradient(ag.w(z))
    u_data = jnp.clip(jnp.asarray(b["noise_preimage"]), -c["u_clip"], c["u_clip"])
    noise = jax.random.normal(jax.random.PRNGKey(77), (obs.shape[0], DA))
    base_key = jax.random.PRNGKey(88)
    return obs, goals, w_rows, u_data, noise, base_key


# --------------------------------------------------------------------- default (distill)

def test_default_objective_is_distill_and_unchanged():
    """actor_objective defaults to distill; a train_actor update on that path is finite, moves
    the actor and reports actor_q -- the existing DSRL behaviour, unchanged."""
    assert get_config()["actor_objective"] == "distill"
    ag = _agent(train_actor=True)
    assert ag.config["actor_objective"] == "distill"
    new, info = ag.update(_batch())
    assert jnp.isfinite(info["actor_q"])
    assert _changed(ag.actor.params, new.actor.params)
    # distill info shape: no awr/bc keys leak into the default path
    assert "awr_weight_mean" not in info and "actor_bc_logp" not in info


# --------------------------------------------------------------------- awr weights

def test_awr_weight_matches_hand_computation():
    """weight_i == clip(exp((Q_data_i - V_i)/beta), 0, wmax), recomputed independently."""
    ag = _agent(train_actor=True, train_basis=False, actor_objective="awr")
    b = _batch(seed=1)
    obs, goals, w_rows, u_data, _, base_key = _actor_inputs(ag, b)
    weight, adv, Q, V = ag._awr_weights(obs, goals, w_rows, u_data, base_key)

    # independent recomputation of Q_data and V from mesh_M with the same base_key
    bp = ag.basis.params
    Q_h = jnp.mean(ag.mesh_M(obs, u_data, goals, w_rows, params=bp), axis=1)
    Kv, scale = int(ag.config["awr_baseline_k"]), float(ag.config["u_clip"])
    u_prior = jnp.clip(jax.random.normal(base_key, (Kv, obs.shape[0], DA)), -scale, scale)
    V_h = jnp.mean(jax.vmap(lambda uk: jnp.mean(
        ag.mesh_M(obs, uk, goals, w_rows, params=bp), axis=1))(u_prior), axis=0)
    beta, wmax = float(ag.config["awr_beta"]), float(ag.config["awr_wmax"])
    w_h = jnp.clip(jnp.exp((Q_h - V_h) / beta), 0.0, wmax)

    assert jnp.allclose(Q, Q_h, atol=1e-5)
    assert jnp.allclose(V, V_h, atol=1e-5)
    assert jnp.allclose(adv, Q_h - V_h, atol=1e-5)
    assert jnp.allclose(weight, w_h, atol=1e-5)
    assert jnp.all(weight >= 0.0) and jnp.all(weight <= wmax + 1e-6)


def test_awr_higher_advantage_gets_higher_weight():
    """Sorting rows by advantage sorts the weights non-decreasing (exp is monotone; clip keeps
    order)."""
    ag = _agent(train_actor=True, train_basis=False, actor_objective="awr")
    b = _batch(seed=2)
    obs, goals, w_rows, u_data, _, base_key = _actor_inputs(ag, b)
    weight, adv, _, _ = ag._awr_weights(obs, goals, w_rows, u_data, base_key)
    order = jnp.argsort(adv)
    w_sorted = weight[order]
    assert jnp.all(jnp.diff(w_sorted) >= -1e-6), "weight must be monotone in advantage"


def test_awr_loss_is_weighted_nll_of_u_data():
    """actor_loss_inloop(awr) == -mean(weight_i * logpi(u_data_i | s_i, w_i)), weights stop-grad."""
    ag = _agent(train_actor=True, train_basis=False, actor_objective="awr")
    b = _batch(seed=3)
    obs, goals, w_rows, u_data, noise, base_key = _actor_inputs(ag, b)
    loss, info = ag.actor_loss_inloop(ag.actor.params, obs, goals, w_rows, noise, u_data, base_key)

    weight, _, _, _ = ag._awr_weights(obs, goals, w_rows, u_data, base_key)
    mu, log_std = ag.actor(obs, w_rows)
    logpi = tanh_gaussian_logprob(mu, log_std, u_data, float(ag.config["u_clip"]))
    hand = -jnp.mean(weight * logpi)
    assert jnp.allclose(loss, hand, atol=1e-5)
    assert jnp.isfinite(loss)
    assert jnp.allclose(info["actor_logp"], jnp.mean(logpi), atol=1e-5)


def test_awr_moves_actor_and_freezes_basis():
    """A full awr update (train_basis=false) moves the actor and leaves the measure frozen."""
    ag = _agent(train_actor=True, train_basis=False, actor_objective="awr")
    new, info = ag.update(_batch(seed=4))
    assert _changed(ag.actor.params, new.actor.params), "actor must move"
    assert _same(ag.basis.params, new.basis.params), "basis frozen"
    assert _same(ag.w.params, new.w.params), "w(z) frozen"
    assert jnp.isfinite(info["awr_weight_mean"]) and jnp.isfinite(info["awr_weight_max"])


# --------------------------------------------------------------------- bc

def test_bc_loss_is_unweighted_nll_of_u_data():
    ag = _agent(train_actor=True, train_basis=False, actor_objective="bc")
    b = _batch(seed=5)
    obs, goals, w_rows, u_data, noise, base_key = _actor_inputs(ag, b)
    loss, info = ag.actor_loss_inloop(ag.actor.params, obs, goals, w_rows, noise, u_data, base_key)
    mu, log_std = ag.actor(obs, w_rows)
    logpi = tanh_gaussian_logprob(mu, log_std, u_data, float(ag.config["u_clip"]))
    assert jnp.allclose(loss, -jnp.mean(logpi), atol=1e-5)
    assert jnp.allclose(info["actor_bc_logp"], jnp.mean(logpi), atol=1e-5)


def test_bc_moves_actor_freezes_basis():
    ag = _agent(train_actor=True, train_basis=False, actor_objective="bc")
    new, _ = ag.update(_batch(seed=6))
    assert _changed(ag.actor.params, new.actor.params)
    assert _same(ag.basis.params, new.basis.params)


# --------------------------------------------------------------------- sfbc acting

def test_sfbc_returns_argmax_actor_candidate_and_is_deterministic():
    """acting=sfbc: candidates come from the actor at eval_w_star; the returned latent is the
    argmax of the goal-averaged measure over them, and it is deterministic given the seed."""
    ag = _agent(train_actor=True, train_basis=False, actor_objective="bc", acting="sfbc")
    b = _batch(reward_rows=(1, 3))
    ag = ag.infer_eval_goals(b, b["rewards"] + 1.0)
    obs0 = jnp.asarray(b["observations"][0])
    seed = jax.random.PRNGKey(2)

    u1 = ag.select_latent_sfbc(obs0, seed)
    u2 = ag.select_latent_sfbc(obs0, seed)
    assert jnp.array_equal(u1, u2), "deterministic given the seed"

    # reconstruct the candidate set from the actor with the same seed
    c = ag.config
    K, scale = int(c["sfbc_num_u"]), float(c["u_clip"])
    obs = jnp.broadcast_to(obs0, (K, obs0.shape[-1]))
    w = jnp.broadcast_to(ag.eval_w_star, (K, ag.eval_w_star.shape[0]))
    mu, log_std = ag.actor(obs, w)
    noise = jax.random.normal(seed, (K, DA))
    u_cand, _ = tanh_gaussian_sample(mu, log_std, noise, scale)
    u_cand = jnp.clip(u_cand, -scale, scale)
    q = jnp.mean(ag.mesh_M(obs, u_cand, ag.eval_goals, w), axis=1)
    expect = u_cand[jnp.argmax(q)]
    assert jnp.allclose(u1, expect, atol=1e-6), "must be the argmax candidate"
    assert jnp.any(jnp.all(jnp.isclose(u_cand, u1[None], atol=1e-6), axis=1)), "one of the candidates"
    assert jnp.all(jnp.abs(u1) <= scale + 1e-6)


def test_sfbc_sample_actions_decodes_clipped_action():
    ag = _agent(train_actor=True, train_basis=False, actor_objective="bc", acting="sfbc")
    b = _batch(reward_rows=(2, 5))
    ag = ag.infer_eval_goals(b, b["rewards"] + 1.0)
    a = ag.sample_actions(jnp.asarray(b["observations"][0]), seed=jax.random.PRNGKey(0))
    assert a.shape == (DA,)
    assert jnp.all(jnp.abs(a) <= 1.0 + 1e-6)


def test_sfbc_infer_eval_goals_does_no_actor_training():
    """acting=sfbc must not re-distill / train the actor at eval; the actor is unchanged."""
    ag = _agent(train_actor=True, train_basis=False, actor_objective="bc", acting="sfbc")
    b = _batch(reward_rows=(1,))
    ev = ag.infer_eval_goals(b, b["rewards"] + 1.0)
    assert _same(ag.actor.params, ev.actor.params), "sfbc eval must not train the actor"
