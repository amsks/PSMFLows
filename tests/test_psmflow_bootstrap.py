"""`bootstrap=gpi_argmax`: the EMaQ-style backup for the measure (2026-09-16).

Under `policy_index=task_vector` the index slot carries w, so "the policy indexed by w" can be
DEFINED as GPI under w: u*(s) = argmax over N prior draws u_m of [mean_P - kappa*unc]
psi(s, w, u_m)^T w. `bootstrap=gpi_argmax` puts that u*(s') -- scored with the TARGET psi and
stop-gradded -- in the backup's action slot, so psi(s, w, u) becomes the successor measure of
"take u, then act by GPI-under-w forever": the policy that is actually deployed. Every
candidate is a clipped prior draw, so the bootstrap decode stays in-support (EMaQ's argument).

What is pinned here:
  - the DEFAULT `bootstrap=index` leaves u_next bit-identical to today's backup;
  - `gpi_argmax` is refused under `policy_index=latent` (the index would not be w);
  - u_next is the pessimistic argmax of the target psi over the N candidates drawn from the
    documented key (`fold_in(rng, 108)`), and lies inside the u box;
  - the whole arm (task_vector, no actor, GPI acting, gpi_argmax backup) updates finitely
    and acts through `gpi_select`'s task-vector branch.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_psmflow_agent import ACT, OBS, _agent, _batch
from utils.psm_common import targets_uncertainty

N = 8
BOOTSTRAP_KEY = 108   # the fold_in id sample_step_inputs uses for the candidate draw


def _emaq_agent(**overrides):
    return _agent(policy_index="task_vector", train_actor=False, acting="gpi",
                  bootstrap="gpi_argmax", bootstrap_candidates=N, **overrides)


def test_default_bootstrap_is_index_and_the_backup_is_unchanged():
    agent = _agent()
    assert agent.config["bootstrap"] == "index"
    s = agent.sample_step_inputs(_batch(), jax.random.PRNGKey(0))
    assert bool(jnp.array_equal(s.u_next, s.u_index)), "default backup must continue u'"


def test_gpi_argmax_is_refused_under_the_latent_index():
    with pytest.raises(AssertionError):
        _agent(policy_index="latent", bootstrap="gpi_argmax")


def _brute_force_argmax(agent, obs, w, key, n):
    """The contract the helper must satisfy, computed the slow way."""
    c = agent.config
    u_cand = jnp.clip(jax.random.normal(key, (n, obs.shape[0], ACT)), -c["u_clip"], c["u_clip"])

    def score(u_m):
        q = (agent.psi_b(obs, w, u_m, params=agent.target_psi) * w).sum(-1)   # (P, B)
        q_mean, q_unc = targets_uncertainty(q, c["num_parallel"])
        return q_mean - c["pessimism_penalty"] * q_unc                          # (B,)

    Q = jax.vmap(score)(u_cand)                                                 # (n, B)
    best = jnp.argmax(Q, axis=0)
    return jnp.take_along_axis(u_cand, best[None, :, None], axis=0)[0], u_cand


def test_u_next_is_the_pessimistic_argmax_of_the_target_psi_over_n_candidates():
    agent = _emaq_agent()
    batch, rng = _batch(), jax.random.PRNGKey(3)
    s = agent.sample_step_inputs(batch, rng)
    key = jax.random.fold_in(rng, BOOTSTRAP_KEY)
    u_star, u_cand = _brute_force_argmax(agent, jnp.asarray(batch["next_observations"]),
                                         s.task_w, key, N)
    np.testing.assert_allclose(np.asarray(s.u_next), np.asarray(u_star), rtol=0, atol=1e-6)
    # every row's bootstrap latent is one of that row's own candidates
    hit = jnp.any(jnp.all(jnp.isclose(u_cand, s.u_next[None]), axis=-1), axis=0)
    assert bool(jnp.all(hit))
    assert np.all(np.abs(np.asarray(s.u_next)) <= agent.config["u_clip"] + 1e-6)


def test_u_next_is_not_the_untrained_actor_latent():
    """Today's task-vector backup reads the (untrained) actor; the new mode must not."""
    agent = _emaq_agent()
    batch, rng = _batch(), jax.random.PRNGKey(0)
    s = agent.sample_step_inputs(batch, rng)
    rs = jax.random.split(rng, 7)
    actor_u = agent._deploy_latent(jnp.asarray(batch["next_observations"]), s.task_w,
                                   jax.random.normal(rs[3], (batch["observations"].shape[0], ACT)))
    assert not np.allclose(np.asarray(s.u_next), np.asarray(actor_u))


def test_emaq_arm_updates_finitely_and_acts_through_gpi():
    agent = _emaq_agent()
    agent, info = agent.update(_batch())
    for k in ["psm_loss", "orth_loss", "bootstrap/adv"]:
        assert math.isfinite(float(info[k])), (k, info[k])
    assert float(info["bootstrap/adv"]) >= 0.0, "advantage of the argmax over the candidate mean"
    a = agent.sample_actions(np.zeros((OBS,), np.float32), seed=jax.random.PRNGKey(1))
    assert a.shape == (ACT,)
    assert np.all(np.abs(np.asarray(a)) <= 1.0 + 1e-6)
