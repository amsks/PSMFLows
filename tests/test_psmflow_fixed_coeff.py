"""`acting=fixed_coeff`: PSM's Eq. 10 test-time policy read through the affine head.

The seam adds one agent field (`fixed_coeff`, the coefficient c in R^{w_dim}) and one
selection rule. Two things have to hold: the defaults do not move (training and the
shipped gpi acting are byte-identical whether or not the field exists), and at c = w(u'_0)
the fixed-coefficient readout IS the single-index readout the pair scan uses.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.psmflow import PSMFlowAgent, load_fixed_index_coeff
from tests.test_psmflow_agent import ACT, OBS, _agent, _batch, _config
from tools.eval_checkpoint import check_fixed_coeff_npz
from utils.psm_common import targets_uncertainty

W_DIM = 8


def _write_coeff(path, c, env_name="cube-single-play-singletask-task2-v0",
                 restore_path="/exp/affine_strict_cube/sd001_x", restore_epoch=500000):
    np.savez(path, c=np.asarray(c, np.float32), env_name=env_name, restore_path=restore_path,
             restore_epoch=restore_epoch)
    return str(path)


def _small_affine(**overrides):
    c = _config(**overrides)
    with c.unlocked():
        c["affine"]["w_dim"] = W_DIM
    return c


def _agent_cfg(cfg):
    return PSMFlowAgent.create(0, np.zeros((1, OBS), np.float32), np.zeros((1, ACT), np.float32), cfg)


def _leaves(agent):
    return jax.tree_util.tree_leaves((agent.phi.params, agent.psi.params, agent.target_phi, agent.target_psi))


def test_default_agent_carries_a_zero_coefficient_and_trains_identically(tmp_path):
    """The default agent never reads `fixed_coeff`; a fixed_coeff agent trains the SAME
    params from the same batches, i.e. the seam touches acting only."""
    rng = np.random.default_rng(0)
    path = _write_coeff(tmp_path / "c.npz", rng.standard_normal(W_DIM))
    base = _agent_cfg(_small_affine())
    fixed = _agent_cfg(_small_affine(acting="fixed_coeff", fixed_index_coeff_path=path))
    assert base.fixed_coeff.shape == (W_DIM,) and float(jnp.abs(base.fixed_coeff).max()) == 0.0
    for i in range(3):
        base, _ = base.update(_batch(i))
        fixed, _ = fixed.update(_batch(i))
    for a, b in zip(_leaves(base), _leaves(fixed)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_fixed_coeff_at_a_family_member_is_the_single_index_readout():
    """c = w(u'_0) makes psi_c = A^T w(u'_0) + beta, so Q_c equals the pair scan's score at
    that one index for every (s, u) -- the identity the acting rule relies on."""
    agent = _agent_cfg(_small_affine())
    rng = np.random.default_rng(1)
    obs = jnp.asarray(rng.standard_normal((16, OBS)), jnp.float32)
    u = jnp.asarray(rng.standard_normal((16, ACT)), jnp.float32)
    u0 = jnp.asarray(rng.standard_normal((1, ACT)), jnp.float32)
    w = jnp.asarray(rng.standard_normal((16, agent.config["z_dim"])), jnp.float32)
    c = agent.psi(u0, method="encode_index")[0]                                 # (w_dim,)
    q_fixed = agent._psi_q_fixed_coeff(obs, u, w, c)                              # (P, B)
    q_index = agent._psi_q_over_indices(obs, u, w, jnp.broadcast_to(u0, (1, 16, ACT)))[:, 0]
    np.testing.assert_allclose(np.asarray(q_fixed), np.asarray(q_index), rtol=1e-5, atol=1e-5)


def test_fixed_coeff_select_is_the_argmax_of_its_own_pessimistic_score(tmp_path):
    """The selected latent is the candidate with the largest [mean - kappa*unc] Q_c over
    the K draws of the same key split `gpi_select` uses for u_cand."""
    rng = np.random.default_rng(2)
    c = rng.standard_normal(W_DIM).astype(np.float32)
    path = _write_coeff(tmp_path / "c.npz", c)
    agent = _agent_cfg(_small_affine(acting="fixed_coeff", fixed_index_coeff_path=path, gpi_num_u=16))
    agent = agent.replace(task_z=jnp.asarray(rng.standard_normal(agent.config["z_dim"]), jnp.float32))
    np.testing.assert_allclose(np.asarray(agent.fixed_coeff), c)
    obs = jnp.asarray(rng.standard_normal(OBS), jnp.float32)
    key = jax.random.PRNGKey(11)
    u_star = agent.fixed_coeff_select(obs, seed=key)
    r_u, _ = jax.random.split(key)
    K, clip = agent.config["gpi_num_u"], agent.config["u_clip"]
    u_cand = jnp.clip(jax.random.normal(r_u, (K, ACT)), -clip, clip)
    q = agent._psi_q_fixed_coeff(jnp.broadcast_to(obs, (K, OBS)), u_cand,
                                 jnp.broadcast_to(agent.task_z, (K, agent.config["z_dim"])), agent.fixed_coeff)
    qm, qu = targets_uncertainty(q, agent.config["num_parallel"])
    score = qm - agent.config["actor_pessimism_penalty"] * qu
    np.testing.assert_array_equal(np.asarray(u_star), np.asarray(u_cand[jnp.argmax(score)]))
    assert float(jnp.abs(u_star).max()) <= clip
    # and sample_actions deploys exactly that latent's decode
    a = agent.sample_actions(obs, seed=key, temperature=0.0)
    np.testing.assert_allclose(np.asarray(a), np.asarray(agent.decode(obs[None], u_star[None])[0]),
                               rtol=1e-6, atol=1e-6)


def test_fixed_coeff_refuses_missing_path_wrong_head_and_wrong_width(tmp_path):
    with pytest.raises(AssertionError, match="fixed_index_coeff_path"):
        _agent_cfg(_small_affine(acting="fixed_coeff"))
    path = _write_coeff(tmp_path / "c.npz", np.ones(W_DIM))
    with pytest.raises(AssertionError, match="psi_form=affine"):
        _agent_cfg(_small_affine(acting="fixed_coeff", fixed_index_coeff_path=path, psi_form="free",
                                 policy_index="task_vector", train_actor=True))
    with pytest.raises(AssertionError, match="w_dim"):
        load_fixed_index_coeff(path, W_DIM + 1)
    with pytest.raises(AssertionError, match="acting"):
        _agent(acting="nonsense")


def test_eval_guard_matches_env_and_checkpoint(tmp_path):
    path = _write_coeff(tmp_path / "c.npz", np.ones(W_DIM))
    meta = check_fixed_coeff_npz(path, "cube-single-play-singletask-task2-v0",
                                 "/somewhere/else/affine_strict_cube/sd001_x", 500000)
    assert meta["env_name"] == "cube-single-play-singletask-task2-v0"
    with pytest.raises(AssertionError, match="fitted for"):
        check_fixed_coeff_npz(path, "cube-single-play-singletask-task3-v0", "/x/sd001_x", 500000)
    with pytest.raises(AssertionError, match="checkpoint"):
        check_fixed_coeff_npz(path, "cube-single-play-singletask-task2-v0", "/x/sd000_y", 500000)
    with pytest.raises(AssertionError, match="epoch"):
        check_fixed_coeff_npz(path, "cube-single-play-singletask-task2-v0", "/x/sd001_x", 250000)
