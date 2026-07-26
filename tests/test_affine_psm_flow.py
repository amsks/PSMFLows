"""Flow (flowBC) actor path for the affine PSM agent.

The reference (../Factored-FB) uses the flow actor for cube/scene/puzzle and the
deterministic actor only for antmaze/locomotion, so `affine_psm` gains the same
`actor.type: ddpgbc | flow` switch that `agents/psm.py` already has. These tests pin the
flow branch's construction, gradient flow, acting path, and the ddpgbc branch's survival.
"""
import math

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf

from agents import agents


def _config(actor_type="flow"):
    with initialize(version_base="1.3", config_path="../configs/agent"):
        cfg = compose(config_name="affine_psm")
    config = ml_collections.ConfigDict(OmegaConf.to_container(cfg, resolve=True))
    config["actor"]["type"] = actor_type
    # Small nets keep the B^2 mesh + the 10-step ODE rollout cheap on CPU.
    config["measure"]["hidden_dim"] = 32
    config["measure"]["hidden_layers"] = 2
    config["actor"]["flow_actor_hidden_dim"] = 32
    config["actor"]["flow_vf_hidden_dim"] = 32
    config["actor"]["flow_vf_hidden_layers"] = 2
    config["actor"]["hidden_dim"] = 32
    return config


def _batch(n=8, obs=8, act=2):
    rng = np.random.default_rng(0)
    return dict(
        observations=rng.standard_normal((n, obs)).astype(np.float32),
        actions=np.clip(rng.standard_normal((n, act)), -1, 1).astype(np.float32),
        next_observations=rng.standard_normal((n, obs)).astype(np.float32),
        index=np.arange(n, dtype=np.int64),
        masks=np.ones((n,), np.float32),
    )


def _agent(actor_type="flow"):
    config = _config(actor_type)
    config["batch_size"] = 8
    return agents["affine_psm"].create(0, np.zeros((1, 8), np.float32),
                                       np.zeros((1, 2), np.float32), config)


def test_flow_actor_builds_a_velocity_field():
    agent = _agent("flow")
    assert agent.config["actor_type"] == "flow"
    assert agent.actor_vf is not None, "flow actor must own a velocity field TrainState"


def test_ddpgbc_actor_has_no_velocity_field():
    agent = _agent("ddpgbc")
    assert agent.config["actor_type"] == "ddpgbc"
    assert agent.actor_vf is None


def test_flow_update_runs_and_reports_both_losses():
    agent = _agent("flow")
    agent, info = agent.update(_batch())
    # bc_flow_loss (velocity field) and actor_bc (one-step distillation) are the two
    # quantities that only exist on the flow path.
    for k in ["psm_loss", "actor_loss", "bc_flow_loss", "actor_bc", "actor_q"]:
        assert k in info, f"missing {k}"
        assert math.isfinite(float(info[k])), (k, info[k])


def test_flow_update_moves_both_actor_and_vf_params():
    agent = _agent("flow")
    before_actor, before_vf = agent.actor.params, agent.actor_vf.params
    agent, _ = agent.update(_batch())

    def _moved(a, b):
        return max(float(jnp.abs(x - y).max())
                   for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))

    assert _moved(before_actor, agent.actor.params) > 0, "actor params did not move"
    assert _moved(before_vf, agent.actor_vf.params) > 0, "velocity field params did not move"


def test_flow_velocity_field_learns_the_data_actions():
    # The flow-matching term is a plain regression onto (a - u0), so it must fall on a
    # fixed batch. This is the piece that makes the distillation target a behaviour
    # SAMPLE rather than the mode-averaging conditional mean.
    agent = _agent("flow")
    b = _batch()
    losses = []
    for _ in range(60):
        agent, info = agent.update(b)
        losses.append(float(info["bc_flow_loss"]))
    assert np.mean(losses[-10:]) < np.mean(losses[:10]), losses[:3] + losses[-3:]


def test_flow_sample_actions_shape_and_range():
    agent = _agent("flow")
    obs = np.zeros((5, 8), np.float32)
    a = agent.sample_actions(obs, seed=jax.random.PRNGKey(0))
    assert a.shape == (5, 2)
    assert float(jnp.abs(a).max()) <= 1.0, "flow actor output must be clipped to [-1, 1]"


def test_flow_sample_actions_varies_with_the_noise_seed():
    # A one-step noise actor is stochastic by construction: distinct seeds must give
    # distinct actions, otherwise the noise input is being ignored.
    agent = _agent("flow")
    obs = np.zeros((5, 8), np.float32)
    a0 = agent.sample_actions(obs, seed=jax.random.PRNGKey(0))
    a1 = agent.sample_actions(obs, seed=jax.random.PRNGKey(1))
    assert float(jnp.abs(a0 - a1).max()) > 1e-6


def test_ddpgbc_path_still_trains():
    agent = _agent("ddpgbc")
    agent, info = agent.update(_batch())
    assert "bc_flow_loss" not in info
    for k in ["psm_loss", "actor_loss", "actor_q"]:
        assert math.isfinite(float(info[k])), (k, info[k])


def test_total_loss_dispatches_on_actor_type():
    for actor_type, expect_flow in (("flow", True), ("ddpgbc", False)):
        agent = _agent(actor_type)
        _, info = agent.total_loss(_batch())
        assert ("bc_flow_loss" in info) is expect_flow, actor_type
