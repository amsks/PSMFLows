"""Tests for hindsight-skill conditioning in utils/flow_inversion.py.

Stage A landed hindsight-skill conditioning: the flow becomes G(s, c, u), with `agent
config skill_cond`/`skill_window`, and `agent._actor_obs(observations, skills)`
concatenates `skills` onto `observations` INTERNALLY inside `compute_flow_actions` /
`sample_actions` whenever `skill_cond` is on (agents/fql.py). `skills` lives in
observation space (same shape as `observations` -- see `utils.datasets.add_skill_targets`),
never a separate lower-dim code.

`_get_preimage_and_jacobian` (the backward-ODE inverter) was NOT given a `skills` kwarg,
so it still needs the concat done externally to match the actor nets' widened input; that
external concat is `_with_skills`, used here (only) ahead of the inverter, while forward
calls thread `skills` through as a kwarg the way agents/fql.py now expects. These tests
characterize that split against tiny real FQLAgents, mirroring the conventions of
tests/test_flow_inversion.py and tests/test_preimage_pipeline.py.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.fql import FQLAgent, get_config
from utils.flow_inversion import (
    augment_dataset_with_point_preimage,
    augment_dataset_with_preimage_distribution,
)

OBS, ACT = 4, 2


@pytest.fixture(autouse=True)
def _force_float32():
    """The BC-flow inversion scan is not x64-safe; see tests/test_flow_inversion.py."""
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


def _tiny_agent(ob_dim, act_dim=ACT, flow_steps=50, skill_cond=False):
    """Minimal real FQLAgent; small hidden dims keep the CPU round-trip fast.

    `ob_dim` is always the RAW observation width: when `skill_cond`, `FQLAgent.create`
    widens the actor nets' example input itself (agents/fql.py), so the example
    observations passed in here are never pre-concatenated.
    """
    cfg = get_config()
    cfg['flow_steps'] = flow_steps
    cfg['actor_hidden_dims'] = (32, 32)
    cfg['value_hidden_dims'] = (32, 32)
    cfg['skill_cond'] = skill_cond
    ex_obs = jnp.zeros((1, ob_dim))
    ex_act = jnp.zeros((1, act_dim))
    return FQLAgent.create(0, ex_obs, ex_act, cfg)


def test_skills_none_matches_direct_agent_calls():
    """Regression pin: skills=None must byte-match today's unconditioned API exactly."""
    n_steps = 20
    agent = _tiny_agent(ob_dim=OBS, flow_steps=n_steps)
    rng = np.random.default_rng(0)
    N = 6
    obs = rng.standard_normal((N, OBS)).astype(np.float32)
    act = np.clip(rng.standard_normal((N, ACT)), -1, 1).astype(np.float32)
    ds = dict(observations=obs, actions=act)
    cfg = dict(n_initial_steps=n_steps, batch_size=4)

    out_omitted = augment_dataset_with_point_preimage(agent, dict(ds), cfg)
    out_explicit = augment_dataset_with_point_preimage(agent, dict(ds), cfg, skills=None)

    # Independently reproduce the pre-existing (pre-skills) computation directly against
    # the agent, with no concat in the loop, to pin behavior against the underlying API
    # rather than against this module's own (possibly buggy) refactor.
    x0_direct = np.asarray(jax.jit(jax.vmap(
        lambda s, a: agent._get_preimage_and_jacobian(s, a, n_steps)[0]
    ))(jnp.asarray(obs), jnp.asarray(act)))
    recon_direct = np.asarray(agent.compute_flow_actions(jnp.asarray(obs), noises=jnp.asarray(x0_direct)))
    roundtrip_direct = np.linalg.norm(recon_direct - np.clip(act, -1, 1), axis=-1)

    np.testing.assert_allclose(out_omitted['noise_preimage_point'], x0_direct, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out_omitted['preimage_roundtrip'], roundtrip_direct, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(out_omitted['noise_preimage_point'], out_explicit['noise_preimage_point'])
    np.testing.assert_array_equal(out_omitted['preimage_roundtrip'], out_explicit['preimage_roundtrip'])


def test_skills_none_matches_mixture_api_too():
    """Same regression pin for the EM-mixture entry point."""
    agent = _tiny_agent(ob_dim=OBS)
    N = 10
    rng = np.random.default_rng(1)
    ds = dict(
        observations=rng.standard_normal((N, OBS)).astype(np.float32),
        actions=np.clip(rng.standard_normal((N, ACT)), -1, 1).astype(np.float32),
    )
    cfg = dict(num_clusters=2, alpha=1.0, num_samples=6, n_steps=1, n_initial_steps=2,
               batch_size=8, seed=0)

    out_omitted = augment_dataset_with_preimage_distribution(agent, dict(ds), cfg)
    out_explicit = augment_dataset_with_preimage_distribution(agent, dict(ds), cfg, skills=None)

    for key in ('noise_preimage_mean', 'noise_preimage_cov', 'noise_preimage_weights', 'preimage_ess'):
        np.testing.assert_array_equal(out_omitted[key], out_explicit[key])


def test_skills_conditioning_round_trips_on_a_toy_conditioned_flow():
    """Inverting G([s; c], u) = a with the correct `c` recovers the generating latent.

    Build a REAL `skill_cond=True` toy agent (Stage A widens the actor nets' input to
    2 * ob_dim itself), generate actions by running the FORWARD flow -- through the
    agent's own `skills=` kwarg, exactly as a skill-conditioned rollout would -- from a
    known noise, then invert through this module's `skills=` and check the recovered
    latent matches the noise that produced the action almost exactly.
    """
    n_steps = 50
    agent = _tiny_agent(ob_dim=OBS, flow_steps=n_steps, skill_cond=True)
    rng = np.random.default_rng(3)
    N = 8
    # Kept away from the [-1, 1] clip (see agent's compute_flow_actions): a clipped action
    # is not invertible, since many latents map to the same clipped boundary value.
    obs = (0.3 * rng.standard_normal((N, OBS))).astype(np.float32)
    skills = (0.3 * rng.standard_normal((N, OBS))).astype(np.float32)  # skills live in obs space
    true_noise = (0.3 * rng.standard_normal((N, ACT))).astype(np.float32)

    actions = np.asarray(agent.compute_flow_actions(
        jnp.asarray(obs), noises=jnp.asarray(true_noise), skills=jnp.asarray(skills)))
    assert np.all(np.abs(actions) < 0.999), 'test actions saturated the clip; scale down the noise'

    ds = dict(observations=obs, actions=actions)
    cfg = dict(n_initial_steps=n_steps, batch_size=8)
    aug = augment_dataset_with_point_preimage(agent, ds, cfg, skills=skills)

    err = np.linalg.norm(aug['noise_preimage_point'] - true_noise, axis=-1)
    assert np.all(err < 1e-4), f'round-trip with the correct skills too high: {err}'
    assert np.all(np.isfinite(aug['preimage_roundtrip']))


def test_wrong_skills_degrade_the_roundtrip():
    """The wrong skills must recover a DIFFERENT latent -- proof conditioning is live.

    Same toy setup as above, but invert against skills that were NOT used to generate the
    action. If `_with_skills` were a no-op (or skills were silently dropped somewhere in
    the pipeline), this would recover the same latent as the correct-skills case; instead
    the wrong conditioning defines a materially different flow, so it recovers a stale,
    much less accurate preimage of the true noise.
    """
    n_steps = 50
    agent = _tiny_agent(ob_dim=OBS, flow_steps=n_steps, skill_cond=True)
    rng = np.random.default_rng(3)
    N = 8
    obs = (0.3 * rng.standard_normal((N, OBS))).astype(np.float32)
    skills = (0.3 * rng.standard_normal((N, OBS))).astype(np.float32)
    true_noise = (0.3 * rng.standard_normal((N, ACT))).astype(np.float32)

    actions = np.asarray(agent.compute_flow_actions(
        jnp.asarray(obs), noises=jnp.asarray(true_noise), skills=jnp.asarray(skills)))

    ds = dict(observations=obs, actions=actions)
    cfg = dict(n_initial_steps=n_steps, batch_size=8)

    aug_correct = augment_dataset_with_point_preimage(agent, ds, cfg, skills=skills)
    err_correct = np.linalg.norm(aug_correct['noise_preimage_point'] - true_noise, axis=-1)

    wrong_skills = (0.3 * rng.standard_normal((N, OBS))).astype(np.float32)
    aug_wrong = augment_dataset_with_point_preimage(agent, ds, cfg, skills=wrong_skills)
    err_wrong = np.linalg.norm(aug_wrong['noise_preimage_point'] - true_noise, axis=-1)

    assert np.mean(err_wrong) > 100 * np.mean(err_correct), (
        f'wrong skills did not degrade recovery of the generating latent '
        f'(correct mean {np.mean(err_correct)}, wrong mean {np.mean(err_wrong)}); '
        'conditioning may not be live in the inversion'
    )


def test_em_mixture_path_threads_skills():
    """The EM entry point must run (not TypeError) on a skill_cond agent and actually
    condition: identical rows with different skills must get different mixtures. Pins
    the agents/fql.py skills= threading through compute_full_proposal_distribution_em."""
    agent = _tiny_agent(ob_dim=OBS, skill_cond=True)
    N = 6
    rng = np.random.default_rng(2)
    obs = np.tile(rng.standard_normal((1, OBS)).astype(np.float32), (N, 1))
    acts = np.tile(np.clip(rng.standard_normal((1, ACT)), -1, 1).astype(np.float32), (N, 1))
    skills = rng.standard_normal((N, OBS)).astype(np.float32)
    ds = dict(observations=obs, actions=acts)
    cfg = dict(num_clusters=1, alpha=1.0, num_samples=6, n_steps=1, n_initial_steps=2,
               batch_size=8, seed=0)
    out = augment_dataset_with_preimage_distribution(agent, dict(ds), cfg, skills=skills)
    means = np.asarray(out['noise_preimage_mean'])
    assert np.all(np.isfinite(means))
    assert not np.allclose(means[0], means[1]), "skills do not reach the EM inversion"
