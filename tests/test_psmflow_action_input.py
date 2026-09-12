"""Action-conditioned affine PSMFlow: exact data actions and decoded latent queries."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.psmflow import PSMFlowAgent, get_config, targets_uncertainty

OBS, ACT, B = 5, 2, 4


def _config(**overrides):
    config = get_config()
    with config.unlocked():
        config.allow_untrained_flow = True
        config.flow_ckpt_path = None
        config.z_dim = 4
        config.gpi_num_u = 3
        config.index_panel = 3
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
        config.actor.flow_steps = 1
        config.q_dist.hidden_dim = 8
        config.q_dist.hidden_layers = 1
        config.dsrl_na.hidden_dim = 8
        config.dsrl_na.hidden_layers = 1
        config.dsrl_na.inner_steps = 1
        config.action_critic.hidden_dim = 8
        config.action_critic.hidden_layers = 1
        config.action_critic.embedding_layers = 2
        config.action_critic.spread_candidates = 2
        config.residual.hidden_dim = 8
        config.residual.hidden_layers = 1
        config.residual.embedding_layers = 2
        config.flow.hidden_dims = (8,)
        config.flow.value_hidden_dims = (8,)
        for key, value in overrides.items():
            config[key] = value
    return config


def _agent(**overrides):
    return PSMFlowAgent.create(
        0,
        np.zeros((1, OBS), np.float32),
        np.zeros((1, ACT), np.float32),
        _config(**overrides),
    )


def _batch(seed=0):
    rng = np.random.default_rng(seed)
    return {
        'observations': rng.standard_normal((B, OBS)).astype(np.float32),
        'actions': np.clip(rng.standard_normal((B, ACT)), -1, 1).astype(np.float32),
        'next_observations': rng.standard_normal((B, OBS)).astype(np.float32),
        'noise_preimage': rng.standard_normal((B, ACT)).astype(np.float32),
    }


def _tree_allclose(left, right, **kwargs):
    return all(
        np.allclose(np.asarray(x), np.asarray(y), **kwargs)
        for x, y in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right))
    )


def _tree_l2(tree):
    return math.sqrt(sum(float(jnp.vdot(x, x)) for x in jax.tree_util.tree_leaves(tree)))


def test_action_queries_equal_the_raw_head_at_decoded_actions():
    agent = _agent(measure_action_input='action')
    rng = np.random.default_rng(1)
    obs = rng.standard_normal((B, OBS)).astype(np.float32)
    index = rng.standard_normal((B, ACT)).astype(np.float32)
    u = rng.standard_normal((B, ACT)).astype(np.float32)

    decoded = agent.decode(obs, u)
    want = agent.bound_psi(agent.psi(obs, index, decoded, params=agent.target_psi))
    got = agent.psi_b(obs, index, u, params=agent.target_psi)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-6, atol=1e-6)


def test_latent_mode_queries_the_raw_head_at_the_latent_exactly():
    agent = _agent(measure_action_input='latent')
    rng = np.random.default_rng(2)
    obs = rng.standard_normal((B, OBS)).astype(np.float32)
    index = rng.standard_normal((B, ACT)).astype(np.float32)
    u = rng.standard_normal((B, ACT)).astype(np.float32)

    want = agent.bound_psi(agent.psi(obs, index, u))
    np.testing.assert_array_equal(np.asarray(agent.psi_b(obs, index, u)), np.asarray(want))


def test_latents_with_the_same_decode_have_the_same_action_conditioned_psi():
    agent = _agent(measure_action_input='action')
    zero_flow = jax.tree_util.tree_map(jnp.zeros_like, agent.flow_onestep)
    agent = agent.replace(flow_onestep=zero_flow)
    obs = np.zeros((B, OBS), np.float32)
    index = np.full((B, ACT), 0.25, np.float32)
    u1 = np.full((B, ACT), -0.75, np.float32)
    u2 = np.full((B, ACT), 0.75, np.float32)

    np.testing.assert_array_equal(np.asarray(agent.decode(obs, u1)), np.asarray(agent.decode(obs, u2)))
    np.testing.assert_allclose(
        np.asarray(agent.psi_b(obs, index, u1)),
        np.asarray(agent.psi_b(obs, index, u2)),
        rtol=0,
        atol=0,
    )


def test_action_measure_step_uses_actions_and_ignores_cached_preimages():
    agent = _agent(measure_action_input='action')
    batch = _batch(3)
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(7))

    changed_u = sampled.replace(u_data=sampled.u_data + 1.25)
    step_a, _ = agent.apply_update(batch, sampled)
    step_b, _ = agent.apply_update(batch, changed_u)
    assert _tree_allclose(step_a.psi.params, step_b.psi.params, rtol=0, atol=0)
    assert _tree_allclose(step_a.phi.params, step_b.phi.params, rtol=0, atol=0)

    changed_actions = dict(batch)
    changed_actions['actions'] = np.asarray(batch['actions']) * -0.5 + 0.2
    step_c, _ = agent.apply_update(changed_actions, sampled)
    assert not _tree_allclose(step_a.psi.params, step_c.psi.params, rtol=1e-7, atol=1e-7)


def test_action_affine_index_panel_matches_explicit_scores():
    agent = _agent(measure_action_input='action')
    rng = np.random.default_rng(4)
    obs = rng.standard_normal((B, OBS)).astype(np.float32)
    u = rng.standard_normal((B, ACT)).astype(np.float32)
    task_w = rng.standard_normal((B, agent.config['z_dim'])).astype(np.float32)
    indices = rng.standard_normal((3, B, ACT)).astype(np.float32)

    got = agent._psi_q_over_indices(obs, u, task_w, indices)
    want = jnp.stack([(agent.psi_b(obs, index, u) * task_w).sum(-1) for index in indices], axis=1)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=2e-5, atol=2e-5)


def test_action_gpi_factorization_preserves_pair_order_and_first_tie():
    agent = _agent(measure_action_input='action')
    task_w = jnp.asarray([0.4, -0.2, 0.7, 0.1], jnp.float32)
    agent = agent.replace(task_z=task_w)
    observation = jnp.asarray([0.1, -0.3, 0.5, 0.2, -0.4], jnp.float32)
    seed = jax.random.PRNGKey(19)
    K, d_a = agent.config['gpi_num_u'], ACT

    r_u, r_index = jax.random.split(seed)
    u = jnp.clip(jax.random.normal(r_u, (K, d_a)), -agent.config['u_clip'], agent.config['u_clip'])
    index = jnp.clip(
        jax.random.normal(r_index, (K, d_a)), -agent._index_clip(), agent._index_clip()
    )
    u_pairs = jnp.repeat(u, K, axis=0)
    index_pairs = jnp.tile(index, (K, 1))
    obs_pairs = jnp.broadcast_to(observation, (K * K, OBS))
    q_ens = (agent.psi_b(obs_pairs, index_pairs, u_pairs) * task_w).sum(-1)
    q_mean, q_unc = targets_uncertainty(q_ens, agent.config['num_parallel'])
    score = q_mean - agent.config['actor_pessimism_penalty'] * q_unc
    want = u_pairs[jnp.argmax(score)]
    np.testing.assert_array_equal(np.asarray(agent.gpi_select(observation, seed)), np.asarray(want))

    tied = agent.replace(task_z=jnp.zeros_like(task_w))
    np.testing.assert_array_equal(np.asarray(tied.gpi_select(observation, seed)), np.asarray(u[0]))


def test_action_actor_gradient_runs_through_frozen_decoder_and_update_is_finite():
    actor_cfg = _config()['actor'].to_dict()
    actor_cfg['bc_coeff'] = 0.0
    agent = _agent(
        measure_action_input='action', train_actor=True, acting='actor', actor=actor_cfg
    )
    batch = _batch(5)
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(8))

    grad = jax.grad(lambda params: agent.flow_actor_loss(
        batch, sampled, params, agent.actor_vf.params
    )[0])(agent.actor.params)
    assert math.isfinite(_tree_l2(grad)) and _tree_l2(grad) > 0.0

    zero_flow = jax.tree_util.tree_map(jnp.zeros_like, agent.flow_onestep)
    flat_agent = agent.replace(flow_onestep=zero_flow)
    flat_grad = jax.grad(lambda params: flat_agent.flow_actor_loss(
        batch, sampled, params, flat_agent.actor_vf.params
    )[0])(flat_agent.actor.params)
    assert _tree_l2(flat_grad) == 0.0

    before_flow = (agent.flow_vf, agent.flow_onestep)
    stepped, info = agent.apply_update(batch, sampled)
    assert math.isfinite(float(info['psm_loss'])) and math.isfinite(float(info['actor_loss']))
    assert not _tree_allclose(agent.actor.params, stepped.actor.params, rtol=0, atol=0)
    assert _tree_allclose(before_flow, (stepped.flow_vf, stepped.flow_onestep), rtol=0, atol=0)


def test_measure_action_input_defaults_backfill_and_guards():
    assert get_config()['measure_action_input'] == 'latent'

    old = _config()
    del old['measure_action_input']
    restored = PSMFlowAgent.create(
        0, np.zeros((1, OBS), np.float32), np.zeros((1, ACT), np.float32), old
    )
    assert restored.config['measure_action_input'] == 'latent'

    with pytest.raises(AssertionError, match='measure_action_input'):
        _agent(measure_action_input='unknown')
    with pytest.raises(AssertionError, match='requires psi_form=affine'):
        _agent(measure_action_input='action', psi_form='free')
    with pytest.raises(AssertionError, match='requires policy_index=latent'):
        _agent(measure_action_input='action', psi_form='affine', policy_index='task_vector')
    with pytest.raises(AssertionError, match='measure_u_samples=1'):
        _agent(
            measure_action_input='action', measure_u_samples=2,
            measure_u_source='jitter', measure_u_jitter_std=0.1,
        )
