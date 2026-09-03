"""E3 Arm B: psi indexed by a POLICY LATENT, as the write-up's Sec. PSMFlows states it.

The shipped agent trains psi(s, w, u): the index slot carries the task vector, and the TD
backup bootstraps the amortized actor's latent at s'. The write-up proves things about
psi(s, u, u') -- a fresh u' ~ p0 per batch element indexes the policy, the backup
continues that SAME u' at s' (so the bootstrap action G(s', u') is a p0 decode, which is
Prop. insample's hypothesis and the only way its C=1 applies), and w enters only through
the readout Q = psi^T w.

What is pinned here:
  - the default path is untouched, bit for bit (published runs stay reproducible);
  - psi's index slot really changes width, from z_dim to d_a;
  - the backup index equals the online index -- psibar(s', u', u'), not the actor's latent;
  - a gradient step runs and is finite with the actor branch removed entirely;
  - acting searches (u_i, u'_j) pairs and returns a latent inside the clip box;
  - acting=actor with train_actor=false is refused rather than silently deploying an
    untrained actor.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

import pytest

from tests.test_psmflow_agent import ACT, _agent, _batch


def _armb(**overrides):
    return _agent(policy_index="latent", train_actor=False, acting="gpi", **overrides)


def test_default_config_is_the_shipped_agent():
    agent = _agent()
    assert agent.config["policy_index"] == "task_vector"
    assert agent.config["train_actor"] is True
    s = agent.sample_step_inputs(_batch(), jax.random.PRNGKey(0))
    assert s.u_index is None, "the index draw must not exist on the default path"
    assert agent._index(s) is s.task_w


def test_index_slot_is_the_latent_and_has_action_width():
    agent = _armb()
    s = agent.sample_step_inputs(_batch(), jax.random.PRNGKey(0))
    assert s.u_index is not None
    assert s.u_index.shape[-1] == ACT, "the index is a latent, not a z_dim task vector"
    assert agent._index(s) is s.u_index
    clip = agent.config["u_clip"]
    assert np.all(np.abs(np.asarray(s.u_index)) <= clip + 1e-6), "index draw must be clipped"
    # psi's first-layer input width follows the slot it is initialized against.
    kernels = jax.tree_util.tree_leaves(agent.psi.params)
    assert any(k.shape[-2] == ACT or ACT in k.shape for k in kernels if k.ndim >= 2)


def test_backup_continues_the_same_index():
    """psibar is read at (s', u', u'): the continuation is pi_{u'}, not the actor."""
    agent = _armb()
    s = agent.sample_step_inputs(_batch(), jax.random.PRNGKey(0))
    assert bool(jnp.array_equal(s.u_next, s.u_index))
    # ... and it is NOT the actor's latent, which is what the shipped backup uses.
    base = _agent().sample_step_inputs(_batch(), jax.random.PRNGKey(0))
    assert not bool(jnp.array_equal(s.u_next, base.u_next))


def test_backup_exploration_is_inert_under_the_latent_index():
    """The bootstrap action is already a p0 decode, so the explore knob cannot change it."""
    a = _armb().sample_step_inputs(_batch(), jax.random.PRNGKey(0))
    b = _armb(backup_explore_frac=1.0).sample_step_inputs(_batch(), jax.random.PRNGKey(0))
    assert bool(jnp.array_equal(a.u_next, b.u_next))


def test_update_runs_without_an_actor():
    agent = _armb()
    before = jax.tree_util.tree_leaves(agent.actor.params)
    agent2, info = agent.update(_batch())
    for k in ("psm_loss", "orth_loss"):
        assert math.isfinite(float(info[k])), (k, info[k])
    assert "actor_loss" not in info, "train_actor=false must not run the actor branch"
    after = jax.tree_util.tree_leaves(agent2.actor.params)
    assert all(bool(jnp.array_equal(x, y)) for x, y in zip(before, after)), \
        "the actor must not move when its branch is off"


def test_gpi_searches_pairs_and_stays_in_the_box():
    agent = _armb(gpi_num_u=8)
    obs = _batch()["observations"][0]
    u = agent.gpi_select(obs, seed=jax.random.PRNGKey(3))
    assert u.shape == (ACT,)
    clip = agent.config["u_clip"]
    assert np.all(np.abs(np.asarray(u)) <= clip + 1e-6)
    # The selected latent is one of the K action candidates, not one of the K indices:
    # Alg. Rung 1 returns G(s, u_ihat).
    r_u, _ = jax.random.split(jax.random.PRNGKey(3))
    cand = np.asarray(jnp.clip(jax.random.normal(r_u, (8, ACT)), -clip, clip))
    assert np.isclose(cand, np.asarray(u)).all(-1).any()


def test_acting_actor_without_an_actor_is_refused():
    agent = _agent(policy_index="latent", train_actor=False, acting="actor")
    with pytest.raises(AssertionError, match="untrained actor"):
        agent.sample_actions(_batch()["observations"][0], seed=jax.random.PRNGKey(0))


def test_latent_index_with_a_trained_actor_takes_a_finite_step():
    """policy_index='latent' + train_actor=true: psi's index slot is the POLICY LATENT.

    `flow_actor_loss` used to hardcode the task vector `w` in psi's index slot, so this
    combination raised ScopeParamShapeError on the very first update (the slot is d_a
    wide, not z_dim) -- and nothing asserted against it, leaving a hole in Arm B's guard
    rail. The coherent semantics, and what the code now does: the actor stays
    w-conditioned (it is the policy the deployed action comes from), the index slot
    carries u' exactly as `measure_loss` and `gpi_select` read it, and the readout stays
    Q = psi(s, u', u_a)^T w.
    """
    agent = _agent(policy_index="latent", train_actor=True, acting="gpi")
    agent2, info = agent.update(_batch())
    for k in ("psm_loss", "orth_loss", "actor_loss", "actor_q", "actor_bc_error"):
        assert k in info, f"{k} missing -- the actor branch did not run"
        assert math.isfinite(float(info[k])), (k, info[k])
    assert not all(bool(jnp.array_equal(x, y)) for x, y in
                   zip(jax.tree_util.tree_leaves(agent.actor.params),
                       jax.tree_util.tree_leaves(agent2.actor.params))), "actor did not train"


def test_actor_loss_reads_psi_at_the_policy_index_not_the_task_vector():
    """The index the actor's Q is read at is `_index(sampled)`, i.e. u' under Arm B."""
    agent = _agent(policy_index="latent", train_actor=True, acting="gpi")
    s = agent.sample_step_inputs(_batch(), jax.random.PRNGKey(0))
    assert agent._index(s) is s.u_index
    # A z_dim-wide task vector in that slot cannot even be evaluated: this is the exact
    # failure the fix removes, and it is what pins the slot to the latent.
    obs = _batch()["observations"]
    u_a = agent.config["u_clip"] * agent.actor(obs, s.task_w, s.flow_noise)
    ok = agent.psi(obs, s.u_index, u_a)          # the fixed call
    assert np.isfinite(np.asarray(ok)).all()
    with pytest.raises(Exception):               # noqa: B017 -- flax raises ScopeParamShapeError
        agent.psi(obs, s.task_w, u_a)            # the old, hardcoded-w call
