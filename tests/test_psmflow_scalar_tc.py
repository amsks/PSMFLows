"""Fix 1 (2026-09-15): the task-conditioned scalar critic on PSM's reward, three seams.

  phi_restore_path / phi_restore_epoch   phi's params from another run's checkpoint, into
                                         phi and target_phi; with train_phi=false they
                                         never move.
  dsrl_na.ignore_masks                   qa's TD target with mask = 1 on every row.
  dsrl_na.reward_scale                   multiplies the synthetic reward phi(s')^T w.

Every default is the OFF value, so the arms before this date restore unchanged. Run
module-per-process: `pytest tests/test_psmflow_scalar_tc.py`.
"""
import os
import pickle

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_psmflow_agent import _agent, _batch
from tests.test_psmflow_dsrl_na import NA_CFG, _na_agent, _na_batch


def _same(left, right):
    for x, y in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        if not np.array_equal(np.asarray(x), np.asarray(y)):
            return False
    return True


def _save_phi(agent, run_dir, epoch):
    """A checkpoint in `save_agent`'s layout carrying only what `_load_phi_params` reads."""
    os.makedirs(run_dir, exist_ok=True)
    state = {"agent": {"phi": {"params": flax.serialization.to_state_dict(agent.phi.params)}}}
    with open(os.path.join(run_dir, f"params_{epoch}.pkl"), "wb") as f:
        pickle.dump(state, f)


# ------------------------------------------------------------------ phi restore
def test_phi_restore_loads_that_checkpoints_phi_into_phi_and_target_phi(tmp_path):
    source = _agent()
    for i in range(2):
        source, _ = source.update(_batch(i))          # move phi off its init
    run_dir = str(tmp_path / "source_run")
    _save_phi(source, run_dir, 7)

    fresh = _agent()
    assert not _same(fresh.phi.params, source.phi.params)
    agent = _agent(phi_restore_path=run_dir, phi_restore_epoch=7)
    assert _same(agent.phi.params, source.phi.params)
    assert _same(agent.target_phi, source.phi.params)
    # only phi is read from the checkpoint: psi starts at the same init as a fresh agent
    assert _same(agent.psi.params, fresh.psi.params)


def test_restored_phi_stays_unchanged_under_train_phi_false(tmp_path):
    source = _agent()
    source, _ = source.update(_batch(0))
    run_dir = str(tmp_path / "source_run")
    _save_phi(source, run_dir, 3)

    agent = _agent(phi_restore_path=run_dir, phi_restore_epoch=3, train_phi=False)
    psi_before = agent.psi.params
    for i in range(3):
        agent, info = agent.update(_batch(i))
    assert _same(agent.phi.params, source.phi.params)
    assert _same(agent.target_phi, source.phi.params)
    assert not _same(agent.psi.params, psi_before)
    assert np.isfinite(float(info["psm_loss"]))


def test_restored_phi_trains_when_train_phi_is_left_on(tmp_path):
    source = _agent()
    run_dir = str(tmp_path / "source_run")
    _save_phi(source, run_dir, 1)
    agent = _agent(phi_restore_path=run_dir, phi_restore_epoch=1)
    agent, _ = agent.update(_batch(0))
    assert not _same(agent.phi.params, source.phi.params)


def test_phi_restore_refuses_a_mismatched_basis(tmp_path):
    source = _agent(z_dim=8)
    run_dir = str(tmp_path / "source_run")
    _save_phi(source, run_dir, 1)
    with pytest.raises(AssertionError, match="shape"):
        _agent(phi_restore_path=run_dir, phi_restore_epoch=1)


def test_phi_restore_refuses_an_ambiguous_glob(tmp_path):
    for name in ("a", "b"):
        _save_phi(_agent(), str(tmp_path / name), 1)
    with pytest.raises(AssertionError, match="exactly one"):
        _agent(phi_restore_path=str(tmp_path / "*"), phi_restore_epoch=1)


def test_phi_restore_composes_with_the_scalar_tc_arm(tmp_path):
    """The Fix 1 arm: D2 (synthetic_w, task-conditioned) on a restored, frozen phi."""
    source = _agent()
    source, _ = source.update(_batch(0))
    run_dir = str(tmp_path / "source_run")
    _save_phi(source, run_dir, 5)
    na = {**NA_CFG, "reward_source": "synthetic_w", "task_conditioned": True,
          "ignore_masks": True, "reward_scale": 0.05}
    agent = _na_agent(dsrl_na=na, phi_restore_path=run_dir, phi_restore_epoch=5,
                      train_phi=False)
    assert _same(agent.phi.params, source.phi.params)
    qa_before = agent.qa.params
    for i in range(3):
        agent, info = agent.update(_na_batch(i))
    assert _same(agent.phi.params, source.phi.params)
    assert not _same(agent.qa.params, qa_before)
    for k in ("na_qa_loss", "na_qw_loss", "na_rw_std", "actor_q"):
        assert np.isfinite(float(info[k])), (k, info[k])


# ------------------------------------------------------------------ ignore_masks
def _qa_target(agent, batch, sampled, mask):
    """r + gamma * mask * min_e qa_bar(s', G(s', u')) for the real-reward arm."""
    na = agent.config["dsrl_na"]
    w = agent._actor_w(sampled.task_w)
    u_next = agent._deploy_latent(batch["next_observations"], w, sampled.flow_noise)
    a_next = agent.decode(batch["next_observations"], u_next)
    q_next = agent.qa(batch["next_observations"], agent._na_in(a_next, w),
                      params=agent.target_qa).min(0)
    return batch["rewards"] + na["discount"] * mask * q_next


def test_ignore_masks_defaults_off_and_reads_the_dataset_masks():
    agent = _na_agent()
    assert agent.config["dsrl_na"]["ignore_masks"] is False
    batch = _na_batch()
    assert 0.0 in set(np.asarray(batch["masks"]).tolist())
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(3))
    _, info = agent.dsrl_qa_loss(batch, sampled, agent.qa.params)
    ref = _qa_target(agent, batch, sampled, batch["masks"])
    np.testing.assert_allclose(float(info["na_qa_target"]), float(ref.mean()), rtol=1e-5)


def test_ignore_masks_makes_the_target_mask_all_ones():
    agent = _na_agent(dsrl_na={**NA_CFG, "ignore_masks": True})
    batch = _na_batch()
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(3))
    loss, info = agent.dsrl_qa_loss(batch, sampled, agent.qa.params)
    ones = jnp.ones_like(batch["masks"])
    ref = _qa_target(agent, batch, sampled, ones)
    q = agent.qa(batch["observations"], batch["actions"], params=agent.qa.params)
    np.testing.assert_allclose(float(loss), float(jnp.square(q - ref[None]).mean()), rtol=1e-5)
    np.testing.assert_allclose(float(info["na_qa_target"]), float(ref.mean()), rtol=1e-5)
    # ... and the dataset masks no longer reach the target at all
    zeroed = {**batch, "masks": np.zeros_like(batch["masks"])}
    l2, _ = agent.dsrl_qa_loss(zeroed, sampled, agent.qa.params)
    np.testing.assert_allclose(float(loss), float(l2), rtol=1e-6)


# ------------------------------------------------------------------ reward_scale
def _d2(**na_over):
    na = {**NA_CFG, "reward_source": "synthetic_w", "task_conditioned": True, **na_over}
    return _na_agent(dsrl_na=na)


def test_reward_scale_multiplies_the_synthetic_reward():
    base, scaled = _d2(), _d2(reward_scale=2.5)
    assert base.config["dsrl_na"]["reward_scale"] == 1.0
    batch = _na_batch()
    key = jax.random.PRNGKey(3)
    s_base = base.sample_step_inputs(batch, key)
    s_scaled = scaled.sample_step_inputs(batch, key)
    _, i_base = base.dsrl_qa_loss(batch, s_base, base.qa.params)
    loss, i_scaled = scaled.dsrl_qa_loss(batch, s_scaled, scaled.qa.params)
    np.testing.assert_allclose(float(i_scaled["na_rw_std"]), 2.5 * float(i_base["na_rw_std"]),
                               rtol=1e-5)
    # the whole target, hand-rolled
    na = scaled.config["dsrl_na"]
    w = s_scaled.task_w
    r_w = 2.5 * (scaled.phi(batch["next_observations"]) * w).sum(-1)
    u_n = scaled._deploy_latent(batch["next_observations"], w, s_scaled.flow_noise)
    a_n = scaled.decode(batch["next_observations"], u_n)
    q_n = scaled.qa(batch["next_observations"], scaled._na_in(a_n, w),
                    params=scaled.target_qa).min(0)
    ref_t = r_w + na["discount"] * batch["masks"] * q_n
    q = scaled.qa(batch["observations"], scaled._na_in(batch["actions"], w),
                  params=scaled.qa.params)
    np.testing.assert_allclose(float(loss), float(jnp.square(q - ref_t[None]).mean()), rtol=1e-5)


def test_reward_scale_leaves_the_real_reward_alone():
    """The multiplier is on the SYNTHETIC reward only; `real` ignores it."""
    batch = _na_batch()
    key = jax.random.PRNGKey(3)
    a, b = _na_agent(), _na_agent(dsrl_na={**NA_CFG, "reward_scale": 7.0})
    la, _ = a.dsrl_qa_loss(batch, a.sample_step_inputs(batch, key), a.qa.params)
    lb, _ = b.dsrl_qa_loss(batch, b.sample_step_inputs(batch, key), b.qa.params)
    np.testing.assert_allclose(float(la), float(lb), rtol=1e-6)
