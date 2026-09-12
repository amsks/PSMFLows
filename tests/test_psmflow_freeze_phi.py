"""Frozen-phi diagnostic switch for paired Stage-C continuations."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from agents.psmflow import PSMFlowAgent, get_config

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBS, ACT, BATCH = 5, 2, 8


def _small_config(**overrides):
    config = get_config()
    with config.unlocked():
        config.allow_untrained_flow = True
        config.flow_ckpt_path = None
        config.z_dim = 4
        config.phi.hidden_dim = 8
        config.phi.hidden_layers = 1
        config.sf.hidden_dim = 8
        config.sf.hidden_layers = 1
        config.sf.embedding_layers = 2
        config.affine.w_dim = 5
        config.affine.encoder_hidden = 8
        config.affine.encoder_layers = 1
        config.actor.hidden_dim = 8
        config.actor.hidden_layers = 1
        config.actor.embedding_layers = 2
        config.actor.vf_hidden_dim = 8
        config.actor.vf_hidden_layers = 1
        config.q_dist.hidden_dim = 8
        config.q_dist.hidden_layers = 1
        config.dsrl_na.hidden_dim = 8
        config.dsrl_na.hidden_layers = 1
        config.action_critic.hidden_dim = 8
        config.action_critic.hidden_layers = 1
        config.action_critic.embedding_layers = 2
        config.residual.hidden_dim = 8
        config.residual.hidden_layers = 1
        config.residual.embedding_layers = 2
        config.flow.hidden_dims = (8,)
        config.flow.value_hidden_dims = (8,)
        for key, value in overrides.items():
            config[key] = value
    return config


def _create(config):
    return PSMFlowAgent.create(
        0,
        np.zeros((1, OBS), np.float32),
        np.zeros((1, ACT), np.float32),
        config,
    )


def _batch(seed=0):
    rng = np.random.default_rng(seed)
    return {
        'observations': rng.standard_normal((BATCH, OBS)).astype(np.float32),
        'actions': np.clip(rng.standard_normal((BATCH, ACT)), -1, 1).astype(np.float32),
        'next_observations': rng.standard_normal((BATCH, OBS)).astype(np.float32),
        'noise_preimage': rng.standard_normal((BATCH, ACT)).astype(np.float32),
    }


def _tree_equal(left, right):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    if len(left_leaves) != len(right_leaves):
        return False
    return all(
        np.array_equal(np.asarray(x), np.asarray(y))
        for x, y in zip(left_leaves, right_leaves)
    )


def _tree_finite(tree):
    return all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(tree))


def test_train_phi_defaults_on_and_matches_the_hydra_config():
    """A run without the diagnostic override must retain the existing training path."""
    with open(os.path.join(REPO, 'configs', 'agent', 'psmflow.yaml')) as fh:
        yaml_config = yaml.safe_load(fh)

    assert get_config()['train_phi'] is True
    assert yaml_config['train_phi'] is True


def test_config_without_train_phi_backfills_to_existing_behavior():
    """Archived configs missing the new key must continue to train phi."""
    config = _small_config()
    with config.unlocked():
        del config['train_phi']

    agent = _create(config)

    assert agent.config['train_phi'] is True


def test_frozen_phi_preserves_full_train_state_while_affine_psi_updates():
    """Freezing must preserve params, Adam state, and step while psi and w(u') still learn."""
    agent = _create(_small_config(train_phi=False))
    phi_before = agent.phi
    psi_before = agent.psi
    w_encoder_before = agent.psi.params['w_enc']

    for seed in range(3):
        agent, _ = agent.update(_batch(seed))

    assert _tree_equal(agent.phi, phi_before)
    assert _tree_equal(agent.target_phi, agent.phi.params)
    assert _tree_finite(agent.psi)
    assert not _tree_equal(agent.psi, psi_before)
    assert not _tree_equal(agent.psi.params['w_enc'], w_encoder_before)


def test_frozen_measure_update_uses_online_phi_instead_of_a_stale_target():
    """A lagging restored target must neither affect psi's step nor survive the update."""
    agent = _create(_small_config(train_phi=False))
    stale_target = jax.tree_util.tree_map(
        lambda x: x + jnp.asarray(0.125, dtype=x.dtype), agent.target_phi)
    stale_agent = agent.replace(target_phi=stale_target)
    batch = _batch(7)
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(9))

    synced_after, _ = agent.apply_update(batch, sampled)
    stale_after, _ = stale_agent.apply_update(batch, sampled)

    assert _tree_equal(stale_after.psi, synced_after.psi)
    assert _tree_equal(stale_after.target_phi, stale_after.phi.params)


def test_train_phi_true_retains_the_existing_joint_update():
    """The default and an explicit true switch must produce the same nontrivial phi step."""
    default_agent = _create(_small_config())
    explicit_agent = _create(_small_config(train_phi=True))
    before = default_agent.phi
    batch = _batch(11)

    default_after, _ = default_agent.update(batch)
    explicit_after, _ = explicit_agent.update(batch)

    assert not _tree_equal(default_after.phi, before)
    assert _tree_equal(default_after.phi, explicit_after.phi)
    assert _tree_equal(default_after.psi, explicit_after.psi)
    assert _tree_equal(default_after.target_phi, explicit_after.target_phi)
