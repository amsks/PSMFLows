"""In-loop FB-style actor+measure interleaving for psmgoal.

These pin the `train_actor` / `train_basis` gates added 2026-09-21:
- default (both flags off-equivalent, i.e. train_actor=false train_basis=true) steps the
  measure and leaves the actor untouched, exactly as the RLU core did;
- train_actor=true supplies the TD backup u' from the actor (not the proto draw) and takes an
  actor gradient step against the measure;
- train_basis=false freezes phi, b and w(z) (the whole measure) so only the actor moves.

Run in its own process: JAX_PLATFORMS=cpu .venv/bin/python -m pytest
tests/test_psmgoal_actor_interleave.py -q -p no:cacheprovider
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from agents.psmgoal import PSMGoalAgent, get_config

OB, DA, Z, CODE, N = 6, 3, 8, 4, 8


def _cfg(**over):
    c = get_config()
    c.z_dim = Z
    c.max_log_seed = CODE
    c.batch_size = N
    c.k_goals = 4
    c.gpi_num_u = 5
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


def _batch(n=N, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "observations": rng.standard_normal((n, OB)).astype(np.float32),
        "next_observations": rng.standard_normal((n, OB)).astype(np.float32),
        "noise_preimage": rng.standard_normal((n, DA)).astype(np.float32),
        "index": np.arange(n).astype(np.int32),
        "rewards": -np.ones((n,), np.float32),
    }


def _changed(tree0, tree1):
    return any(not jnp.array_equal(a, b) for a, b in zip(
        jax.tree_util.tree_leaves(tree0), jax.tree_util.tree_leaves(tree1)))


def _same(tree0, tree1):
    return all(jnp.array_equal(a, b) for a, b in zip(
        jax.tree_util.tree_leaves(tree0), jax.tree_util.tree_leaves(tree1)))


# ------------------------------------------------------- default (actor off, basis on)

def test_default_leaves_actor_frozen_moves_basis():
    """train_actor=false, train_basis=true (defaults): the actor param-tree is unchanged and
    the basis param-tree moves -- the RLU-core update, unchanged."""
    ag = _agent()
    assert ag.config["train_actor"] is False and ag.config["train_basis"] is True
    new, info = ag.update(_batch())
    assert _same(ag.actor.params, new.actor.params), "actor must not move when train_actor=false"
    assert _changed(ag.basis.params, new.basis.params), "basis must move when train_basis=true"
    assert jnp.isfinite(info["psm_loss"])


# ------------------------------------------------------- train_actor: backup + actor step

def test_train_actor_backup_is_actor_not_proto():
    """train_actor=true: the TD backup u' equals the actor's target-coefficient draw, and it
    differs from the proto draw the default path uses."""
    ag = _agent(train_actor=True)
    b = _batch()
    z = jnp.zeros((N, CODE), jnp.float32).at[:, 0].set(1.0)
    u_sel = ag._select_u_next(b, z)
    w_next = ag.w(z, params=ag.target_w)
    u_actor = ag.actor_backup(jnp.asarray(b["next_observations"]), w_next)
    u_proto = ag.proto_bootstrap(z, jnp.asarray(b["index"]))
    assert jnp.allclose(u_sel, u_actor, atol=1e-6), "train_actor backup must be the actor draw"
    assert not jnp.allclose(u_sel, u_proto), "the actor draw must differ from the proto draw"
    assert jnp.all(jnp.abs(u_sel) <= ag.config["u_clip"] + 1e-6)


def test_train_actor_steps_actor():
    """train_actor=true: one update moves the actor param-tree."""
    ag = _agent(train_actor=True)
    new, info = ag.update(_batch())
    assert _changed(ag.actor.params, new.actor.params), "actor must move when train_actor=true"
    assert jnp.isfinite(info["actor_q"])


# ------------------------------------------------------- train_basis=false: frozen measure

def test_train_basis_false_freezes_measure_actor_still_moves():
    """train_basis=false: phi, b (basis) and w(z) are bitwise unchanged and their Polyak
    targets do not move; the actor still steps when train_actor=true."""
    ag = _agent(train_actor=True, train_basis=False)
    new, info = ag.update(_batch())
    assert _same(ag.basis.params, new.basis.params), "basis (phi, b) must be frozen"
    assert _same(ag.w.params, new.w.params), "w(z) must be frozen with the basis"
    assert _same(ag.target_basis, new.target_basis), "target basis must not move"
    assert _same(ag.target_w, new.target_w), "target w must not move"
    assert _changed(ag.actor.params, new.actor.params), "actor must still move"
    assert jnp.isfinite(info["psm_loss"])


def test_full_update_frozen_basis_is_finite():
    """A full train_actor=true, train_basis=false update runs, is finite, moves the actor and
    keeps the basis frozen."""
    ag = _agent(train_actor=True, train_basis=False)
    new, info = ag.update(_batch(seed=3))
    assert all(jnp.isfinite(x).all() for x in jax.tree_util.tree_leaves(new.actor.params))
    assert jnp.isfinite(info["psm_loss"]) and jnp.isfinite(info["actor_q"])
    assert _changed(ag.actor.params, new.actor.params)
    assert _same(ag.basis.params, new.basis.params)
