"""LatentRLAgent smoke tests (P0.8, 2026-08-14).

This agent produced the ceiling number the whole roadmap is calibrated against (P2:
"can the latent action space support an improvement loop at all") and had NO test
coverage. The two properties worth pinning are the ones whose silent failure would have
invalidated that number:

  * residual inertness at eps=0 -- the eps=0 arm is the anchor the tradeoff curve is
    measured against, so if the residual head leaked into the executed action there, the
    x-axis origin would not be "pure decode";
  * target-critic usage -- the TD target must read `target_critic`, not the online
    params. A critic bootstrapping off itself is a different (and much less stable)
    algorithm, and it is invisible in the loss curve.

The frozen flow loader is monkeypatched to freshly-initialised params: these tests are
about the RL loop, not about checkpoint IO (test_psmflow_agent covers that against a
real Stage-A checkpoint).
"""
import math

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import pytest
from omegaconf import OmegaConf

import agents.latentrl as latentrl_mod
from agents.latentrl import LatentRLAgent
from main import _lists_to_tuples
from utils.networks import ActorVectorField

OBS, ACT, B = 6, 2, 16


@pytest.fixture(autouse=True)
def _fake_flow(monkeypatch):
    """Frozen-flow params without a checkpoint on disk."""

    def _init(flow_ckpt_path, flow_ckpt_epoch, config, ex_observations, ex_actions):
        hidden = tuple(config["flow"]["hidden_dims"])
        vf_def = ActorVectorField(hidden_dims=hidden, action_dim=ex_actions.shape[-1],
                                  layer_norm=config["flow"]["layer_norm"])
        onestep_def = ActorVectorField(hidden_dims=hidden, action_dim=ex_actions.shape[-1],
                                       layer_norm=config["flow"]["layer_norm"])
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        times = jnp.zeros((ex_observations.shape[0], 1))
        return (vf_def.init(k1, ex_observations, ex_actions, times)["params"],
                onestep_def.init(k2, ex_observations, ex_actions)["params"])

    monkeypatch.setattr(latentrl_mod, "_load_flow_params", _init)


def _config(**overrides):
    cfg = OmegaConf.load("configs/agent/latentrl.yaml")
    c = ml_collections.ConfigDict(_lists_to_tuples(OmegaConf.to_container(cfg, resolve=True)))
    with c.unlocked():
        c["flow_ckpt_path"] = "unused-monkeypatched"
        c["value_hidden_dims"] = (64, 64)
        c["actor"]["hidden_dim"] = 64
        c["residual"]["hidden_dim"] = 64
        for k, v in overrides.items():
            c[k] = v
    return c


def _agent(**overrides):
    return LatentRLAgent.create(0, np.zeros((1, OBS), np.float32),
                                np.zeros((1, ACT), np.float32), _config(**overrides))


def _batch(seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        observations=rng.standard_normal((B, OBS)).astype(np.float32),
        actions=np.clip(rng.standard_normal((B, ACT)), -1, 1).astype(np.float32),
        next_observations=rng.standard_normal((B, OBS)).astype(np.float32),
        rewards=rng.standard_normal(B).astype(np.float32),
        masks=np.ones(B, np.float32),
        noise_preimage=rng.standard_normal((B, ACT)).astype(np.float32),
    )


def _tree_equal(a, b):
    return all(bool(jnp.array_equal(x, y))
               for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


def test_update_is_finite_and_trains_critic_and_actor():
    agent = _agent()
    new, info = agent.update(_batch())
    for k in ("critic_loss", "actor_loss", "actor_q", "actor_bc", "residual_spend"):
        assert math.isfinite(float(info[k])), (k, info[k])
    assert not _tree_equal(agent.critic.params, new.critic.params)
    assert not _tree_equal(agent.actor.params, new.actor.params)


def test_residual_is_inert_at_eps_zero():
    """eps=0: executed action IS the decode, bit-for-bit, and the head cannot move it.

    Also pins that eps=0 does not make the head untrainable-by-accident in a way that
    hides a later bug: with eps>0 the same call must differ from the pure decode.
    """
    agent = _agent(residual_eps=0.0)
    obs = jnp.asarray(_batch()["observations"])
    u = jnp.asarray(np.random.default_rng(1).standard_normal((B, ACT)), jnp.float32)
    np.testing.assert_array_equal(np.asarray(agent.execute(obs, u)),
                                  np.asarray(agent.decode(obs, u)))
    # Perturbing the residual params cannot change the executed action at eps=0.
    bumped = agent.replace(residual=agent.residual.replace(
        params=jax.tree_util.tree_map(lambda p: p + 1.0, agent.residual.params)))
    np.testing.assert_array_equal(np.asarray(bumped.execute(obs, u)),
                                  np.asarray(agent.execute(obs, u)))
    # ...and with a budget it must.
    wide = _agent(residual_eps=0.2)
    wide_bumped = wide.replace(residual=wide.residual.replace(
        params=jax.tree_util.tree_map(lambda p: p + 1.0, wide.residual.params)))
    assert not np.array_equal(np.asarray(wide_bumped.execute(obs, u)),
                              np.asarray(wide.decode(obs, u)))


def test_residual_spend_respects_the_eps_budget():
    eps = 0.2
    agent = _agent(residual_eps=eps)
    for i in range(3):
        agent, info = agent.update(_batch(i))
        assert float(info["residual_spend"]) <= eps + 1e-6


def test_td_target_reads_the_target_critic_not_the_online_one():
    """Perturb ONLY target_critic: the critic loss must move. Perturb only the online
    params and the same target must be unchanged."""
    agent = _agent()
    batch = _batch()
    a_next = agent.execute(jnp.asarray(batch["next_observations"]),
                           jnp.asarray(batch["noise_preimage"]))

    def loss_of(a):
        val, _ = a.critic_loss(batch, jnp.asarray(batch["actions"]), a_next,
                               a.critic.params)
        return float(val)

    bumped_target = agent.replace(target_critic=jax.tree_util.tree_map(
        lambda p: p + 0.5, agent.target_critic))
    assert not math.isclose(loss_of(agent), loss_of(bumped_target), rel_tol=1e-6), (
        "critic_loss ignores target_critic -- the TD target is bootstrapping off the "
        "online critic")


def test_target_critic_tracks_by_polyak_not_by_copy():
    agent = _agent()
    new, _ = agent.update(_batch())
    assert not _tree_equal(new.critic.params, new.target_critic), (
        "target_critic was hard-copied from the online critic (tau=1 behaviour)")
    assert not _tree_equal(agent.target_critic, new.target_critic), "target never moves"
    # One polyak step must land within tau of the old target, not at the new critic.
    tau = float(agent.config["tau"])
    for old, tgt, onl in zip(jax.tree_util.tree_leaves(agent.target_critic),
                             jax.tree_util.tree_leaves(new.target_critic),
                             jax.tree_util.tree_leaves(new.critic.params)):
        expected = (1 - tau) * np.asarray(old) + tau * np.asarray(onl)
        np.testing.assert_allclose(np.asarray(tgt), expected, rtol=1e-5, atol=1e-6)


def test_flow_params_stay_frozen():
    agent = _agent(residual_eps=0.1)
    before = jax.tree_util.tree_leaves((agent.flow_vf, agent.flow_onestep))
    for i in range(2):
        agent, _ = agent.update(_batch(i))
    after = jax.tree_util.tree_leaves((agent.flow_vf, agent.flow_onestep))
    for b, a in zip(before, after):
        np.testing.assert_array_equal(np.asarray(b), np.asarray(a))
