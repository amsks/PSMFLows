"""f_psmgoal2p: 2P-DSRL on a frozen phi projected from a psmgoal RLUMeasure basis.

The only new code is the basis-load glue: `PsmgoalProjectedPhi` (a drop-in for `PhiMap`
whose params are a frozen `RLUMeasure`) and the `basis_restore_path` branch in
`PSMFlowAgent.create`. Everything downstream (affine psi, dsrl_sac actor, gpi) is the
ordinary agent. These tests pin (1) the projection, (2) the load + freeze, (3) that the
seam is off by default (an ordinary f_psmflow agent is bitwise unchanged), and (4) one full
update step of the arm runs finite with the projected basis frozen.
"""
import math
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.f_psmflow import PSMFlowAgent, get_config
from utils.psm_networks import PhiMap, PsmgoalProjectedPhi, RLUMeasure

OBS, ACT, B = 6, 2, 32
Z = 16                 # small z_dim for CPU
MH, ML = 32, 2         # RLUMeasure hidden_dim / hidden_layers (must match the "checkpoint")
K_U, K_G = 2, 4


# --------------------------------------------------------------------------- (1) module
def _projected_phi(k_u=3, k_g=5, seed=0):
    rng = np.random.default_rng(seed)
    U = np.clip(rng.standard_normal((k_u, ACT)), -3.0, 3.0).astype(np.float32)
    G = rng.standard_normal((k_g, OBS)).astype(np.float32)
    mod = PsmgoalProjectedPhi(z_dim=Z, measure_hidden_dim=MH, measure_hidden_layers=ML,
                              U=jnp.asarray(U), G=jnp.asarray(G))
    params = mod.init(jax.random.PRNGKey(seed), jnp.zeros((1, OBS), np.float32))["params"]
    return mod, params


def test_projected_phi_shape_norm_and_nondegenerate():
    mod, params = _projected_phi()
    # Its only parameters are the wrapped RLUMeasure (submodule `measure`).
    assert set(params.keys()) == {"measure"}
    obs = np.random.default_rng(1).standard_normal((B, OBS)).astype(np.float32)
    f = np.asarray(mod.apply({"params": params}, obs))
    assert f.shape == (B, Z)
    assert np.all(np.isfinite(f))
    # On the sphere of radius sqrt(z_dim), matching PhiMap's project_z.
    np.testing.assert_allclose(np.linalg.norm(f, axis=-1), math.sqrt(Z), rtol=1e-4, atol=1e-4)
    # E[f f^T] is non-degenerate: rows are not all equal and the batch spans > 1 dim.
    assert np.linalg.matrix_rank(f - f.mean(0, keepdims=True)) > 1
    assert f.std(0).mean() > 1e-3


def test_projected_phi_accepts_extra_leading_dims():
    mod, params = _projected_phi()
    obs = np.random.default_rng(2).standard_normal((4, 5, OBS)).astype(np.float32)
    f = np.asarray(mod.apply({"params": params}, obs))
    assert f.shape == (4, 5, Z)
    np.testing.assert_allclose(np.linalg.norm(f, axis=-1), math.sqrt(Z), rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------- config helpers
def _base_config(**overrides):
    c = get_config()
    with c.unlocked():
        c["allow_untrained_flow"] = True    # tests only; real runs require flow_ckpt_path
        c["z_dim"] = Z
        for k, v in overrides.items():
            c[k] = v
    return c


def _twop_config(basis_dir, epoch, goal_states=None, **overrides):
    """The 2P-DSRL arm on a frozen projected psmgoal basis."""
    c = _base_config()
    with c.unlocked():
        c["psi_form"] = "affine"
        c["policy_index"] = "task_vector"
        c["train_actor"] = True
        c["acting"] = "actor"
        c["actor_mode"] = "dsrl_sac"
        c["train_phi"] = False
        c["u_clip"] = 3.0
        c["discount"] = 0.98
        c["actor"]["bc_coeff"] = 0.0
        c["actor"]["target_entropy"] = -3.4657
        c["measure"] = {"hidden_dim": MH, "hidden_layers": ML}
        c["basis_restore_path"] = basis_dir
        c["basis_restore_epoch"] = epoch
        c["basis_k_u"] = K_U
        c["basis_k_g"] = K_G
        c["basis_proj_seed"] = 0
        if goal_states is not None:
            c["basis_goal_states"] = np.asarray(goal_states, np.float32).tolist()
        for k, v in overrides.items():
            c[k] = v
    return c


def _write_basis_checkpoint(tmp_path, epoch=750000, seed=3):
    """A synthetic psmgoal checkpoint: pickle with agent/basis/params = RLUMeasure params."""
    basis = RLUMeasure(z_dim=Z, hidden_dim=MH, hidden_layers=ML)
    params = basis.init(jax.random.PRNGKey(seed),
                        jnp.zeros((1, OBS), np.float32), jnp.zeros((1, ACT), np.float32),
                        jnp.zeros((1, OBS), np.float32))["params"]
    run_dir = os.path.join(str(tmp_path), "psmgoal_run_sd000")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, f"params_{epoch}.pkl"), "wb") as f:
        pickle.dump({"agent": {"basis": {"params": jax.device_get(params)}}}, f)
    return run_dir, epoch, params


def _batch(seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        observations=rng.standard_normal((B, OBS)).astype(np.float32),
        actions=np.clip(rng.standard_normal((B, ACT)), -1, 1).astype(np.float32),
        next_observations=rng.standard_normal((B, OBS)).astype(np.float32),
        noise_preimage=rng.standard_normal((B, ACT)).astype(np.float32),
        rewards=rng.standard_normal((B,)).astype(np.float32),
        masks=np.ones((B,), np.float32),
    )


def _create(config):
    return PSMFlowAgent.create(0, np.zeros((1, OBS), np.float32),
                               np.zeros((1, ACT), np.float32), config)


# --------------------------------------------------------------------------- (2) load + freeze
def test_basis_params_load_into_the_wrapped_measure(tmp_path):
    run_dir, epoch, saved = _write_basis_checkpoint(tmp_path)
    goals = np.random.default_rng(7).standard_normal((K_G, OBS)).astype(np.float32)
    agent = _create(_twop_config(run_dir, epoch, goal_states=goals))
    # phi is the projected basis, and its `measure` subtree equals the checkpoint's params.
    assert isinstance(agent.phi.model_def, PsmgoalProjectedPhi)
    loaded = agent.phi.params["measure"]
    for k in saved:
        for leaf in saved[k]:
            np.testing.assert_allclose(np.asarray(loaded[k][leaf]),
                                       np.asarray(saved[k][leaf]), rtol=1e-6, atol=1e-6)


def test_basis_shape_mismatch_is_rejected(tmp_path):
    # A checkpoint written at a different RLUMeasure width must fail loudly at create().
    run_dir, epoch, _ = _write_basis_checkpoint(tmp_path)
    bad = _twop_config(run_dir, epoch)
    with bad.unlocked():
        bad["measure"] = {"hidden_dim": MH * 2, "hidden_layers": ML}  # wrong width
    with pytest.raises(AssertionError, match="does not match"):
        _create(bad)


def test_projected_basis_is_frozen_across_an_update(tmp_path):
    run_dir, epoch, _ = _write_basis_checkpoint(tmp_path)
    goals = np.random.default_rng(8).standard_normal((K_G, OBS)).astype(np.float32)
    agent = _create(_twop_config(run_dir, epoch, goal_states=goals))
    before = jax.tree_util.tree_leaves(agent.phi.params)
    agent, _ = agent.update(_batch(0))
    after = jax.tree_util.tree_leaves(agent.phi.params)
    for b, a in zip(before, after):
        np.testing.assert_array_equal(np.asarray(b), np.asarray(a))


# --------------------------------------------------------------------------- (3) off by default
def test_seam_is_off_by_default():
    """With no basis_restore_path the agent is an ordinary f_psmflow: phi is a PhiMap and
    its params are identical to a build that never knew the seam existed."""
    a = _create(_base_config())
    b = _create(_base_config())
    assert isinstance(a.phi.model_def, PhiMap)
    assert a.config.get("basis_restore_path", None) is None
    for x, y in zip(jax.tree_util.tree_leaves(a.phi.params),
                    jax.tree_util.tree_leaves(b.phi.params)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    a, info = a.update(_batch(0))
    assert math.isfinite(float(info["psm_loss"]))


# --------------------------------------------------------------------------- (4) full update
def test_full_update_runs_finite_psi_and_actor_move_basis_frozen(tmp_path):
    run_dir, epoch, _ = _write_basis_checkpoint(tmp_path)
    goals = np.random.default_rng(9).standard_normal((K_G, OBS)).astype(np.float32)
    agent = _create(_twop_config(run_dir, epoch, goal_states=goals))

    phi_before = jax.tree_util.tree_leaves(agent.phi.params)
    psi_before = jax.tree_util.tree_leaves(agent.psi.params)
    act_before = jax.tree_util.tree_leaves(agent.sac_actor.params)

    agent, info = agent.update(_batch(0))
    for v in info.values():
        assert np.all(np.isfinite(np.asarray(v))), info

    phi_after = jax.tree_util.tree_leaves(agent.phi.params)
    psi_after = jax.tree_util.tree_leaves(agent.psi.params)
    act_after = jax.tree_util.tree_leaves(agent.sac_actor.params)

    for b, a in zip(phi_before, phi_after):          # projected basis frozen
        np.testing.assert_array_equal(np.asarray(b), np.asarray(a))
    assert any(not np.allclose(np.asarray(b), np.asarray(a))
               for b, a in zip(psi_before, psi_after)), "psi did not move"
    assert any(not np.allclose(np.asarray(b), np.asarray(a))
               for b, a in zip(act_before, act_after)), "dsrl_sac actor did not move"
