"""Fix 2 (2026-09-15): scalar grounding of the readout and the dueling affine head.

  psm_scalar_coef     adds the readout's projected Bellman loss to the measure loss,
                      variance-normalised, gradient to psi only.
  psi_dueling         psi(s,u,z) = V(s,z) + Adv(s,u,z) - mean_k Adv(s,u_k,z) over
                      `psi_dueling_samples` clipped prior latents, inside the affine head.

Pinned: both OFF reproduce the Section 10 affine arm bit for bit against a fixture saved
from the code before this change (`tests/fixtures/psmflow_sec10_affine_3updates.npz`,
3 updates of `_batch(0..2)`); the scalar term against a hand-rolled reference and its
gradient reaches psi only; the dueling baseline averages to zero over its own panel; shapes;
affineness in the policy coordinate survives; both arms train and act. Run
module-per-process: `pytest tests/test_psmflow_scalar_dueling.py`.
"""
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_psmflow_agent import ACT, OBS, _agent, _batch
from utils.psm_common import targets_uncertainty

#: The Section 10 agent on the affine head (docs/design/2026-09-14-flow-psm-dsrl-paper-versions.md).
SEC10 = {"psi_form": "affine", "policy_index": "task_vector",
         "train_actor": True, "acting": "actor"}
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                       "psmflow_sec10_affine_3updates.npz")


def _sec10(**overrides):
    return _agent(**{**SEC10, **overrides})


def _leaves(agent):
    return jax.tree_util.tree_leaves((agent.phi.params, agent.psi.params,
                                      agent.actor.params, agent.actor_vf.params))


def _rand(key, shape):
    return jax.random.normal(jax.random.PRNGKey(key), shape).astype(jnp.float32)


# ------------------------------------------------------------------ off = the published arm
def test_both_seams_default_off():
    agent = _sec10()
    assert float(agent.config["psm_scalar_coef"]) == 0.0
    assert agent.config["psi_dueling"] is False
    assert int(agent.config["psi_dueling_samples"]) == 8


def test_off_reproduces_the_saved_baseline_bit_for_bit():
    """3 updates of the Section 10 affine arm against the fixture saved from the code
    before psm_scalar_coef / psi_dueling existed."""
    saved = np.load(FIXTURE)
    agent = _sec10(psm_scalar_coef=0.0, psi_dueling=False)
    for i in range(3):
        agent, info = agent.update(_batch(i))
    leaves = _leaves(agent)
    assert len(leaves) == len([k for k in saved.files if k.startswith("leaf_")])
    for i, x in enumerate(leaves):
        np.testing.assert_array_equal(np.asarray(x), saved[f"leaf_{i}"])
    assert float(info["psm_loss"]) == float(saved["psm_loss"])
    assert float(info["orth_loss"]) == float(saved["orth_loss"])
    assert float(info["psm_scalar_loss"]) == 0.0


# ------------------------------------------------------------------ scalar grounding
def test_scalar_term_matches_a_hand_rolled_projected_bellman_loss():
    agent, base = _sec10(psm_scalar_coef=1.0), _sec10(psm_scalar_coef=0.0)
    batch = _batch()
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(3))
    loss, info = agent.measure_loss(batch, sampled, agent.phi.params, agent.psi.params)
    loss0, info0 = base.measure_loss(batch, sampled, base.phi.params, base.psi.params)

    c, z = agent.config, sampled.task_w
    obs, next_obs = batch["observations"], batch["next_observations"]
    r_z = (agent.phi(next_obs) * z).sum(-1)                                  # (B,)
    q = (agent.psi(obs, z, sampled.u_data) * z[None]).sum(-1)                # (P, B)
    q_boot = (agent.psi(next_obs, z, sampled.u_next, params=agent.target_psi)
              * z[None]).sum(-1)                                             # (P, B)
    m, unc = targets_uncertainty(q_boot, c["num_parallel"])
    target = r_z + c["discount"] * (m - c["pessimism_penalty"] * unc)
    ref = jnp.mean((q - target[None]) ** 2) / (jnp.var(r_z) + 1e-8)
    np.testing.assert_allclose(float(info["psm_scalar_loss"]), float(ref), rtol=1e-5)
    np.testing.assert_allclose(float(info["psm_scalar_target_std"]), float(target.std()),
                               rtol=1e-5)
    # loss = the published loss + coef * scalar; the measure term itself is untouched
    np.testing.assert_allclose(float(loss - loss0), float(ref), rtol=1e-4)
    assert float(info["psm_loss"]) == float(info0["psm_loss"])


def test_scalar_term_is_weighted_by_the_coefficient():
    a, b = _sec10(psm_scalar_coef=0.5), _sec10(psm_scalar_coef=2.0)
    batch = _batch()
    sampled = a.sample_step_inputs(batch, jax.random.PRNGKey(3))
    la, ia = a.measure_loss(batch, sampled, a.phi.params, a.psi.params)
    lb, ib = b.measure_loss(batch, sampled, b.phi.params, b.psi.params)
    assert float(ia["psm_scalar_loss"]) == float(ib["psm_scalar_loss"])
    np.testing.assert_allclose(float(lb - la), 1.5 * float(ia["psm_scalar_loss"]), rtol=1e-4)


def test_scalar_term_sends_gradient_to_psi_only():
    agent = _sec10(psm_scalar_coef=1.0)
    batch = _batch()
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(3))

    def scalar_only(phi_p, psi_p):
        _, info = agent.measure_loss(batch, sampled, phi_p, psi_p)
        return info["psm_scalar_loss"]

    g_phi, g_psi = jax.grad(scalar_only, argnums=(0, 1))(agent.phi.params, agent.psi.params)
    assert all(np.all(np.asarray(g) == 0) for g in jax.tree_util.tree_leaves(g_phi))
    assert any(np.any(np.asarray(g) != 0) for g in jax.tree_util.tree_leaves(g_psi))


def test_scalar_arm_takes_a_finite_step():
    agent = _sec10(psm_scalar_coef=1.0)
    before = _leaves(agent)
    for i in range(3):
        agent, info = agent.update(_batch(i))
    assert any(not np.array_equal(np.asarray(x), np.asarray(y))
               for x, y in zip(before, _leaves(agent)))
    for k in ("psm_loss", "psm_scalar_loss", "psm_scalar_target_std", "actor_q"):
        assert math.isfinite(float(info[k])), (k, info[k])
    assert float(info["psm_scalar_loss"]) > 0.0


# ------------------------------------------------------------------ dueling head
def test_dueling_requires_the_affine_head():
    with pytest.raises(AssertionError, match="psi_form=affine"):
        _agent(psi_form="free", policy_index="task_vector", train_actor=True,
               acting="actor", psi_dueling=True)


def test_dueling_requires_the_panel_on_every_psi_call():
    agent = _sec10(psi_dueling=True)
    with pytest.raises(AssertionError, match="u_prior"):
        agent.psi(_rand(0, (4, OBS)), _rand(1, (4, 16)), _rand(2, (4, ACT)))


def test_dueling_shapes_and_the_advantage_averages_to_zero_over_its_panel():
    K = 6
    agent = _sec10(psi_dueling=True, psi_dueling_samples=K)
    P = agent.config["num_parallel"]
    obs, z, u = _rand(0, (4, OBS)), _rand(1, (4, 16)), _rand(2, (4, ACT))
    panel = agent._duel_panel(jax.random.PRNGKey(1))
    assert panel.shape == (K, ACT)
    assert float(jnp.abs(panel).max()) <= agent.config["u_clip"]

    out = agent.psi(obs, z, u, u_prior=panel)
    assert out.shape == (P, 4, 16)
    A, beta = agent.psi(obs, u, method="sa_terms", u_prior=panel)
    assert A.shape == (P, 4, 16, agent.config["affine"]["w_dim"]) and beta.shape == (P, 4, 16)
    A_v, beta_v = agent.psi(obs, method="value_terms")
    assert A_v.shape == A.shape and beta_v.shape == beta.shape

    # V(s, z) = beta_V(s) + A_V(s)^T w(z); Adv - baseline averages to ~0 over the panel
    w_z = agent.psi(z, method="encode_index")                              # (4, w_dim)
    V = jnp.einsum("pbzw,bw->pbz", A_v, w_z) + beta_v
    adv = jnp.stack([agent.psi(obs, z, jnp.broadcast_to(u_k, (4, ACT)), u_prior=panel) - V
                     for u_k in panel])                                    # (K, P, 4, 16)
    np.testing.assert_allclose(np.asarray(adv.mean(0)), 0.0, atol=1e-4)
    # ... while the advantage part itself is not identically zero
    assert float(jnp.abs(adv).mean()) > 1e-3
    assert float(jnp.abs(out - V).mean()) > 1e-3


def test_dueling_head_stays_affine_in_the_policy_coordinate():
    agent = _sec10(psi_dueling=True)
    obs, u = _rand(0, (4, OBS)), _rand(2, (4, ACT))
    z1, z2 = _rand(3, (4, 16)), _rand(4, (4, 16))
    panel = agent._duel_panel(jax.random.PRNGKey(2))
    d_psi = agent.psi(obs, z1, u, u_prior=panel) - agent.psi(obs, z2, u, u_prior=panel)
    A, _ = agent.psi(obs, u, method="sa_terms", u_prior=panel)
    d_w = agent.psi(z1, method="encode_index") - agent.psi(z2, method="encode_index")
    np.testing.assert_allclose(np.asarray(d_psi), np.asarray(jnp.einsum("pbzw,bw->pbz", A, d_w)),
                               rtol=1e-4, atol=1e-4)


def test_dueling_and_scalar_arm_trains_and_acts():
    """The launched arm: psm_scalar_coef=1.0 psi_dueling=true on the Section 10 affine agent."""
    agent = _sec10(psi_dueling=True, psm_scalar_coef=1.0)
    before = _leaves(agent)
    for i in range(3):
        agent, info = agent.update(_batch(i))
    assert any(not np.array_equal(np.asarray(x), np.asarray(y))
               for x, y in zip(before, _leaves(agent)))
    for k in ("psm_loss", "orth_loss", "psm_scalar_loss", "actor_q"):
        assert math.isfinite(float(info[k])), (k, info[k])
    b = _batch()
    agent = agent.infer_eval_z(b["next_observations"],
                               np.random.default_rng(1).standard_normal((len(b["observations"]),)))
    a = np.asarray(agent.sample_actions(b["observations"][0], seed=jax.random.PRNGKey(0)))
    assert a.shape == (ACT,) and np.all(np.abs(a) <= 1.0 + 1e-5)


def test_dueling_reaches_the_gpi_scan_of_the_strict_arm():
    """policy_index=latent, acting=gpi: the pair scan passes its own panel through."""
    agent = _agent(psi_dueling=True)
    agent, info = agent.update(_batch())
    assert math.isfinite(float(info["psm_loss"])) and "psi_q_spread_rel" in info
    b = _batch()
    agent = agent.infer_eval_z(b["next_observations"],
                               np.random.default_rng(1).standard_normal((len(b["observations"]),)))
    a = np.asarray(agent.sample_actions(b["observations"][0], seed=jax.random.PRNGKey(0)))
    assert a.shape == (ACT,) and np.all(np.abs(a) <= 1.0 + 1e-5)
