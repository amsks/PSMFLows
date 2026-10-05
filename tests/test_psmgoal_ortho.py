"""F1 of docs/design/2026-09-22-psmgoal-fixes.md: the orthogonality term on psmgoal's phi.

Pins that ortho_coef=0 leaves the loss exactly at the TD mesh loss, that the term is
||mean_ij phi_ij phi_ij^T - I||_F^2 over the same mesh the TD loss reads, that a positive
coefficient adds it with its weight and yields finite gradients, and that the logged
effective rank is the participation ratio of that Gram.

Run this file in its own process (JAX_PLATFORMS=cpu); the whole suite in one process
SIGABRTs in XLA.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from tests.test_psmgoal import _agent, _batch
from utils.psm_proto import sample_z_bin


def _loss(agent, batch):
    z = sample_z_bin(jax.random.PRNGKey(1), batch["observations"].shape[0],
                     int(agent.config["max_log_seed"]))
    u_next = agent._select_u_next(batch, z)
    return agent.measure_loss(agent.basis.params, agent.w.params, batch, z, u_next)


def _gram(agent, batch):
    u = jnp.clip(jnp.asarray(batch["noise_preimage"]), -agent.config["u_clip"], agent.config["u_clip"])
    phi, _ = agent.mesh_phi_b(jnp.asarray(batch["observations"]), u, jnp.asarray(batch["next_observations"]))
    p = np.asarray(phi).reshape(-1, phi.shape[-1]).astype(np.float64)
    return p.T @ p / p.shape[0]


def test_default_ortho_coef_is_zero_and_loss_is_td_only():
    agent, batch = _agent(), _batch()
    assert float(agent.config["ortho_coef"]) == 0.0
    loss, info = _loss(agent, batch)
    np.testing.assert_allclose(float(loss), float(info["psm_offdiag"] + info["psm_diag"]), rtol=1e-6)


def test_ortho_loss_is_gram_deviation_on_the_td_mesh():
    agent, batch = _agent(), _batch()
    _, info = _loss(agent, batch)
    G = _gram(agent, batch)
    expected = np.sum((G - np.eye(G.shape[0])) ** 2)
    np.testing.assert_allclose(float(info["ortho_loss"]), expected, rtol=1e-4)
    pr = np.trace(G) ** 2 / np.sum(G ** 2)
    np.testing.assert_allclose(float(info["phi_eff_rank"]), pr, rtol=1e-4)


def test_positive_coef_adds_weighted_term_and_grads_are_finite():
    batch = _batch()
    a0, a1 = _agent(), _agent(ortho_coef=10.0)
    l0, i0 = _loss(a0, batch)
    l1, _ = _loss(a1, batch)
    np.testing.assert_allclose(float(l1), float(l0) + 10.0 * float(i0["ortho_loss"]), rtol=1e-5)
    new, info = a1.update(batch)
    for leaf in jax.tree_util.tree_leaves(new.basis.params):
        assert np.isfinite(np.asarray(leaf)).all()
    assert np.isfinite(float(info["ortho_loss"]))


def test_ortho_term_lowers_its_own_loss_under_training():
    """A few steps with a large coefficient reduce the Gram deviation it penalises."""
    batch = _batch(n=16)
    agent = _agent(ortho_coef=100.0, batch_size=16, lr_measure=1e-3)
    _, first = _loss(agent, batch)
    for _ in range(30):
        agent, _ = agent.update(batch)
    _, last = _loss(agent, batch)
    assert float(last["ortho_loss"]) < float(first["ortho_loss"])


def test_mesh_M_unchanged_by_refactor():
    """mesh_M is phi^T w + b built from mesh_phi_b."""
    agent, batch = _agent(), _batch()
    obs, g = jnp.asarray(batch["observations"]), jnp.asarray(batch["next_observations"])
    u = jnp.asarray(batch["noise_preimage"])
    w = jax.random.normal(jax.random.PRNGKey(3), (obs.shape[0], int(agent.config["z_dim"])))
    phi, b = agent.mesh_phi_b(obs, u, g)
    np.testing.assert_allclose(np.asarray(agent.mesh_M(obs, u, g, w)),
                               np.asarray((phi * w[:, None, :]).sum(-1) + b), rtol=1e-6)
    # pointwise agreement with M(s,u,g) for the (i, j) = (1, 2) cell
    m12 = agent.M(obs[1:2], u[1:2], g[2:3], w[1:2])
    np.testing.assert_allclose(float(agent.mesh_M(obs, u, g, w)[1, 2]), float(m12[0]), rtol=1e-5)
