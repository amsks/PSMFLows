"""Backup exploration: a fraction of TD bootstrap latents drawn from the prior.

Added 2026-08-10 in response to the flat-Q diagnosis (logs/d3_q_landscape_cube.json): the
backup only ever evaluated psi at the actor's own latent, so psi was fit on the slice of
latent space the actor already occupied and Q came out flat to ~1% of |Q| across the whole
prior.

Two properties, both load-bearing:
  - at the default 0.0 the sampler is BIT-IDENTICAL to the pre-feature code, so every
    published run stays reproducible. This is why the extra keys are split inside the
    branch rather than up front;
  - when enabled, the bootstrap latents genuinely change, and roughly at the requested
    rate.
"""
import jax
import jax.numpy as jnp
import numpy as np

from tests.test_psmflow_agent import _agent, _batch


def _sampled(agent, seed=0):
    return agent.sample_step_inputs(_batch(), jax.random.PRNGKey(seed))


def test_default_is_off_and_matches_actor_latent():
    """frac=0.0 -> u_next is exactly the actor's latent, no RNG stream shift."""
    agent = _agent()
    assert agent.config["backup_explore_frac"] == 0.0
    s = _sampled(agent)

    # Reproduce the bootstrap latent the pre-feature code produced: the actor's output on
    # the key that r_next carried before any extra split existed.
    batch = _batch()
    B, adim = batch["observations"].shape[0], agent.config["action_dim"]
    _, _, _, r_next, _, _, _ = jax.random.split(jax.random.PRNGKey(0), 7)
    expected = agent.config["u_clip"] * agent.actor(
        batch["next_observations"], s.task_w, jax.random.normal(r_next, (B, adim)))
    assert bool(jnp.array_equal(s.u_next, expected)), "default path shifted the RNG stream"


def test_enabled_changes_the_bootstrap_latents():
    off = _sampled(_agent())
    on = _sampled(_agent(backup_explore_frac=0.5))
    assert not bool(jnp.array_equal(off.u_next, on.u_next))
    # Everything else the sampler draws is untouched: only the bootstrap arm differs.
    assert bool(jnp.array_equal(off.task_w, on.task_w))
    assert bool(jnp.array_equal(off.u_data, on.u_data))
    assert bool(jnp.array_equal(off.flow_noise, on.flow_noise))


def test_replacement_rate_is_about_right():
    """~frac of rows differ from the actor's latent, and all stay inside the clip box."""
    frac = 0.5
    agent = _agent(backup_explore_frac=frac)
    base = _sampled(_agent())
    on = _sampled(agent)
    changed = ~np.all(np.isclose(np.asarray(base.u_next), np.asarray(on.u_next)), axis=-1)
    rate = changed.mean()
    # Small batch in the fixture, so this is a sanity band, not a tight test.
    assert 0.2 <= rate <= 0.8, rate
    clip = agent.config["u_clip"]
    assert np.all(np.abs(np.asarray(on.u_next)) <= clip + 1e-6)


def test_update_still_runs_with_exploration_on():
    import math
    agent = _agent(backup_explore_frac=0.3)
    agent2, info = agent.update(_batch())
    for k in ("psm_loss", "orth_loss", "actor_loss"):
        assert math.isfinite(float(info[k])), (k, info[k])


def test_explore_keys_are_folded_not_split_from_a_consumed_key():
    """r_next is the bootstrap-noise SAMPLE key; it must not also be a split parent.

    Pre-2026-09-03 the branch did `jax.random.split(r_next)` after
    `jax.random.normal(r_next, ...)` had already consumed it -- the documented JAX
    anti-pattern. The explore keys now come from `fold_in(rng, 107)`. The default path
    (frac=0) is byte-identical either way, which is what
    `test_default_is_off_and_matches_actor_latent` pins; what is pinned here is that the
    ENABLED branch no longer derives its draws from the consumed key.
    """
    agent = _agent(backup_explore_frac=1.0)
    batch = _batch()
    B, adim = batch["observations"].shape[0], agent.config["action_dim"]
    rng = jax.random.PRNGKey(0)
    s = _sampled(agent)

    clip = agent.config["u_clip"]
    _, _, _, r_next, _, _, _ = jax.random.split(rng, 7)
    old_expl, _ = jax.random.split(r_next)                      # the old, reused parent
    new_expl, _ = jax.random.split(jax.random.fold_in(rng, 107))  # the fixed one
    old_u = jnp.clip(jax.random.normal(old_expl, (B, adim)), -clip, clip)
    new_u = jnp.clip(jax.random.normal(new_expl, (B, adim)), -clip, clip)
    # At frac=1.0 every bootstrap latent is a prior draw, so u_next IS that draw.
    assert bool(jnp.array_equal(s.u_next, new_u)), "explore draw is not folded out of rng"
    assert not bool(jnp.array_equal(s.u_next, old_u)), "still splitting the consumed r_next"
