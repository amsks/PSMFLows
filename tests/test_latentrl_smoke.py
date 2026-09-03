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


# ---------------------------------------------------------------------------
# critic_input=latent -- the offline DSRL-SAC arm (docs/plans/2026-09-03-latentrl-dsrl-sac.md)
# ---------------------------------------------------------------------------

# Two update steps of the DEFAULT (action) path, recorded from the code as it stood before
# `critic_input` existed. The switch's whole claim is that `action` reproduces byte-for-byte
# -- same rng splits, same shapes, same numbers -- so these are pinned, not recomputed.
GOLDEN_ACTION_PATH = {
    "actor_bc": 0.8154887557029724,
    "actor_loss": 9.05768871307373,
    "actor_q": -0.8462899923324585,
    "critic_loss": 4.652002334594727,
    "q_mean": -0.227675199508667,
    "residual_spend": 0.0,
    "target_mean": -1.4628702402114868,
}


def test_action_path_is_bit_for_bit_what_it_was_before_the_switch():
    """critic_input defaults to `action` and that path is the pre-existing computation."""
    agent = _agent()
    assert agent.config["critic_input"] == "action"
    agent, _ = agent.update(_batch(0))
    _, info = agent.update(_batch(1))
    for k, want in GOLDEN_ACTION_PATH.items():
        np.testing.assert_allclose(float(info[k]), want, rtol=1e-6, atol=1e-6,
                                   err_msg=f"{k} moved: the action path is not the old one")
    # ...and asking for it explicitly is the same thing.
    a2 = _agent(critic_input="action")
    a2, _ = a2.update(_batch(0))
    _, info2 = a2.update(_batch(1))
    for k in GOLDEN_ACTION_PATH:
        np.testing.assert_array_equal(np.asarray(info[k]), np.asarray(info2[k]))


def test_latent_critic_takes_a_finite_step():
    agent = _agent(critic_input="latent")
    new, info = agent.update(_batch())
    for k in ("critic_loss", "actor_loss", "actor_q", "bc_loss", "critic_q_data",
              "q_spread", "q_spread_rel"):
        assert k in info, k
        assert math.isfinite(float(info[k])), (k, info[k])
    assert not _tree_equal(agent.critic.params, new.critic.params)
    assert not _tree_equal(agent.actor.params, new.actor.params)


def test_latent_critic_never_touches_the_flow():
    """The point of the arm: no decode anywhere in the training graph.

    Two independent checks -- the frozen params are unchanged after a step (which the
    action path also satisfies), and perturbing the decoder cannot move the actor loss
    (which the action path does NOT satisfy, since its critic scores decoded actions).
    """
    agent = _agent(critic_input="latent")
    before = jax.tree_util.tree_leaves((agent.flow_vf, agent.flow_onestep))
    stepped, _ = agent.update(_batch())
    for b, a in zip(before, jax.tree_util.tree_leaves(
            (stepped.flow_vf, stepped.flow_onestep))):
        np.testing.assert_array_equal(np.asarray(b), np.asarray(a))

    batch = _batch()
    u_data = jnp.clip(jnp.asarray(batch["noise_preimage"]), -3.0, 3.0)
    noise = jnp.asarray(np.random.default_rng(7).standard_normal((B, ACT)), jnp.float32)

    def actor_loss_of(a):
        val, _ = a.actor_loss(batch, u_data, noise, a.actor.params, a.residual.params)
        return float(val)

    bumped = agent.replace(flow_onestep=jax.tree_util.tree_map(
        lambda p: p + 1.0, agent.flow_onestep))
    assert math.isclose(actor_loss_of(agent), actor_loss_of(bumped), rel_tol=1e-6), (
        "the latent actor loss moved when the frozen decoder was perturbed -- there is a "
        "decode in the training graph")
    # The action path is the contrast: there the decoder IS in the actor loss.
    act_agent = _agent(critic_input="action")
    act_bumped = act_agent.replace(flow_onestep=jax.tree_util.tree_map(
        lambda p: p + 1.0, act_agent.flow_onestep))
    assert not math.isclose(actor_loss_of(act_agent), actor_loss_of(act_bumped),
                            rel_tol=1e-6)


def test_latent_critic_scores_latents_not_actions():
    """Q must respond to u and be blind to batch['actions']."""
    agent = _agent(critic_input="latent")
    batch = _batch()
    _, info = agent.update(batch)
    other = dict(batch, actions=-np.asarray(batch["actions"]))
    _, info2 = agent.update(other)
    np.testing.assert_allclose(float(info["critic_loss"]), float(info2["critic_loss"]),
                               rtol=1e-6, atol=1e-7)
    perturbed = dict(batch, noise_preimage=batch["noise_preimage"] + 0.5)
    _, info3 = agent.update(perturbed)
    assert not math.isclose(float(info["critic_loss"]), float(info3["critic_loss"]),
                            rel_tol=1e-6)


def test_latent_actor_bc_weight_is_its_own_knob_and_zero_is_legal():
    """bc_alpha_latent=0 is the pure-DSRL setting: the actor loss stops seeing u_data."""
    batch, moved = _batch(), _batch()
    moved = dict(moved, noise_preimage=moved["noise_preimage"] + 1.0)

    def actor_loss_of(a, b):
        u_data = jnp.clip(jnp.asarray(b["noise_preimage"]), -3.0, 3.0)
        noise = jnp.asarray(np.random.default_rng(7).standard_normal((B, ACT)), jnp.float32)
        val, _ = a.actor_loss(b, u_data, noise, a.actor.params, a.residual.params)
        return float(val)

    a0 = _agent(critic_input="latent", bc_alpha_latent=0.0)
    assert math.isclose(actor_loss_of(a0, batch), actor_loss_of(a0, moved), rel_tol=1e-6), (
        "bc_alpha_latent=0 still anchors to u_data")
    a1 = _agent(critic_input="latent", bc_alpha_latent=10.0)
    assert not math.isclose(actor_loss_of(a1, batch), actor_loss_of(a1, moved),
                            rel_tol=1e-6)
    # ...and the default is the shared number, so the two paths use one alpha unless told.
    assert float(_agent().config["bc_alpha_latent"]) == float(_agent().config["alpha"])
    _, info = a0.update(batch)
    assert math.isfinite(float(info["actor_loss"]))


def test_latent_critic_refuses_a_live_residual():
    with pytest.raises(AssertionError, match="residual_eps"):
        _agent(critic_input="latent", residual_eps=0.05)


def test_q_spread_diagnostic_does_not_disturb_the_training_stream():
    """The spread key is folded out of rng, so the actor/critic draws are unaffected."""
    agent = _agent(critic_input="latent")
    batch = _batch()
    new, info = agent.update(batch)
    # 16 clipped prior draws at up to 64 states: a relative spread, so scale-free.
    assert float(info["q_spread"]) >= 0.0
    # The successor rng must be the plain 3-way split, exactly as the action path's is.
    expected_rng = jax.random.split(agent.rng, 3)[0]
    np.testing.assert_array_equal(np.asarray(new.rng), np.asarray(expected_rng))
