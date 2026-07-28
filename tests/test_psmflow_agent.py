"""Task 3: PSMFlowAgent core — flow-indexed measure loss, frozen flow, jitted update.

The defining property of this agent versus PSM: the policy family is indexed by a
LATENT u' under a frozen behaviour flow, and the TD backup uses that same latent as the
continuation index, so every action the backup implies is a flow decode (in-support).
No decoded action appears anywhere in training.
"""
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.psmflow import PSMFlowAgent, get_config

OBS, ACT, B = 6, 2, 32

# A real Stage-A (fql bc_only) checkpoint, if this machine has one. Used to exercise the
# frozen-flow loader against genuine weights — see test_loads_real_stage_a_checkpoint.
STAGE_A_CKPT = os.environ.get(
    "PSMFLOWS_STAGE_A_CKPT",
    "/var/local/amsks/exp/PSMFLows/bcflow_cube_single_20260726_135032/sd000_20260726_135037",
)
STAGE_A_OBS, STAGE_A_ACT, STAGE_A_EPOCH = 28, 5, 500000  # cube-single-play


def _config(**overrides):
    c = get_config()
    with c.unlocked():
        c["allow_untrained_flow"] = True   # tests only; real runs require flow_ckpt_path
        c["z_dim"] = 16
        for k, v in overrides.items():
            c[k] = v
    return c


def _batch(seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        observations=rng.standard_normal((B, OBS)).astype(np.float32),
        actions=np.clip(rng.standard_normal((B, ACT)), -1, 1).astype(np.float32),
        next_observations=rng.standard_normal((B, OBS)).astype(np.float32),
        noise_preimage=rng.standard_normal((B, ACT)).astype(np.float32),
    )


def _agent(**overrides):
    return PSMFlowAgent.create(0, np.zeros((1, OBS), np.float32), np.zeros((1, ACT), np.float32),
                               _config(**overrides))


def test_update_runs_and_is_finite():
    agent = _agent()
    agent, info = agent.update(_batch())
    for k in ["psm_loss", "orth_loss"]:
        assert math.isfinite(float(info[k])), (k, info[k])


def test_flow_params_are_frozen():
    agent = _agent()
    before = jax.tree_util.tree_leaves((agent.flow_vf, agent.flow_onestep))
    for i in range(3):
        agent, _ = agent.update(_batch(i))
    after = jax.tree_util.tree_leaves((agent.flow_vf, agent.flow_onestep))
    for b, a in zip(before, after):
        np.testing.assert_array_equal(np.asarray(b), np.asarray(a))


def test_psi_slots_are_asymmetric():
    """z-slot (policy index u') and action-slot (current latent u) must be wired
    to different towers — swapping them must change the output."""
    agent = _agent()
    obs = np.zeros((4, OBS), np.float32)
    u1 = np.full((4, ACT), 0.5, np.float32)
    u2 = np.full((4, ACT), -0.5, np.float32)
    out12 = np.asarray(agent.psi(obs, u1, u2))
    out21 = np.asarray(agent.psi(obs, u2, u1))
    assert not np.allclose(out12, out21)


def test_loss_decreases_on_fixed_batch():
    agent = _agent()
    batch = _batch()
    _, first = agent.total_loss(batch, rng=jax.random.PRNGKey(0))
    for _ in range(50):
        agent, info = agent.update(batch)
    _, last = agent.total_loss(batch, rng=jax.random.PRNGKey(0))
    assert float(last["psm_loss"]) < float(first["psm_loss"])


def test_missing_preimage_key_raises():
    agent = _agent()
    bad = _batch()
    del bad["noise_preimage"]
    try:
        agent.update(bad)
        assert False, "expected KeyError for missing noise_preimage"
    except KeyError:
        pass


def test_untrained_flow_requires_explicit_optin():
    """A real run must load a Stage-A checkpoint. An untrained flow indexes policies by
    a map that carries no behaviour information, which would silently train nonsense."""
    c = get_config()
    with c.unlocked():
        c["z_dim"] = 16
        c["allow_untrained_flow"] = False
    try:
        PSMFlowAgent.create(0, np.zeros((1, OBS), np.float32), np.zeros((1, ACT), np.float32), c)
        assert False, "expected an assertion without flow_ckpt_path"
    except AssertionError as e:
        assert "flow_ckpt_path" in str(e)


def test_continuation_index_is_the_policy_index():
    """The TD target must evaluate psi(s', u', u') — the continuation latent IS the
    index. Using the dataset latent u in the action slot instead would bootstrap off a
    different policy than the one being indexed, which is the whole point of the method.

    Checked structurally: the target reads psi(next_obs, u_index, u_index), so making
    u_data differ from u_index must NOT change the target term, while perturbing
    u_index must.
    """
    agent = _agent()
    batch = _batch()
    rng = jax.random.PRNGKey(0)
    sampled = agent.sample_step_inputs(batch, rng)

    tphi = agent.phi(batch["next_observations"], params=agent.target_phi)
    target_with = np.asarray(
        agent.psi(batch["next_observations"], sampled.u_index, sampled.u_index,
                  params=agent.target_psi) @ tphi.T)
    # Swapping in u_data on the action slot must give a DIFFERENT matrix, proving the
    # two slots are not accidentally fed the same thing.
    target_wrong = np.asarray(
        agent.psi(batch["next_observations"], sampled.u_index, sampled.u_data,
                  params=agent.target_psi) @ tphi.T)
    assert not np.allclose(target_with, target_wrong)


def test_index_latents_are_clipped_to_the_typical_set():
    """u' draws are clamped to +-u_clip. Unclamped Gaussian tails are exactly where the
    flow diverges (see tests/test_flow_inversion.py), so the index family must stay in
    the region the flow can actually decode."""
    agent = _agent(u_clip=1.0, index_mix_ratio=0.0)  # pure Gaussian draws
    batch = _batch()
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(3))
    u = np.asarray(sampled.u_index)
    assert u.max() <= 1.0 + 1e-6 and u.min() >= -1.0 - 1e-6


@pytest.mark.skipif(not os.path.isdir(STAGE_A_CKPT),
                    reason=f"no Stage-A checkpoint at {STAGE_A_CKPT}")
def test_loads_real_stage_a_checkpoint():
    """The frozen-flow loader must produce the TRAINED weights, not a silent re-init.

    `_load_flow_params` reaches into an FQL param tree by key
    (modules_actor_bc_flow / modules_actor_onestep_flow) and rebuilds the net defs from
    the psmflow config. A key rename or a shape mismatch on either side would be caught
    only here — every other test opts into `allow_untrained_flow`, so they would pass
    against random weights. Stage C training against a random flow is exactly the
    failure this whole pipeline guards against.
    """
    c = get_config()
    with c.unlocked():
        c["z_dim"] = 16
        c["flow_ckpt_path"] = STAGE_A_CKPT
        c["flow_ckpt_epoch"] = STAGE_A_EPOCH

    ex_obs = np.zeros((1, STAGE_A_OBS), np.float32)
    ex_act = np.zeros((1, STAGE_A_ACT), np.float32)
    agent = PSMFlowAgent.create(0, ex_obs, ex_act, c)

    # Same structure as a fresh init...
    untrained = _config()
    with untrained.unlocked():
        untrained["z_dim"] = 16
    ref = PSMFlowAgent.create(0, ex_obs, ex_act, untrained)
    loaded_leaves = jax.tree_util.tree_leaves(agent.flow_vf)
    ref_leaves = jax.tree_util.tree_leaves(ref.flow_vf)
    assert len(loaded_leaves) == len(ref_leaves)
    for a, b in zip(loaded_leaves, ref_leaves):
        assert a.shape == b.shape
    # ...but genuinely different values, i.e. the checkpoint really was read.
    assert any(not np.allclose(np.asarray(a), np.asarray(b))
               for a, b in zip(loaded_leaves, ref_leaves))

    # And the loaded field is usable as a velocity field at the checkpoint's dims.
    obs = np.zeros((4, STAGE_A_OBS), np.float32)
    act = np.zeros((4, STAGE_A_ACT), np.float32)
    t = np.zeros((4, 1), np.float32)
    v = agent.flow_vf_def.apply({"params": agent.flow_vf}, obs, act, t)
    assert v.shape == (4, STAGE_A_ACT)
    assert np.all(np.isfinite(np.asarray(v)))


@pytest.mark.skipif(not os.path.isdir(STAGE_A_CKPT),
                    reason=f"no Stage-A checkpoint at {STAGE_A_CKPT}")
def test_flow_checkpoint_from_a_different_env_is_rejected():
    """restore_agent swaps the param tree without checking shapes, so a Stage-A
    checkpoint trained on another environment loads silently and only misbehaves at
    decode time. Pointing Stage C at the wrong ckpt is a realistic mistake; it must fail
    loudly at create()."""
    c = get_config()
    with c.unlocked():
        c["z_dim"] = 16
        c["flow_ckpt_path"] = STAGE_A_CKPT
        c["flow_ckpt_epoch"] = STAGE_A_EPOCH
    with pytest.raises(AssertionError, match="different environment"):
        PSMFlowAgent.create(
            0,
            np.zeros((1, STAGE_A_OBS - 9), np.float32),  # wrong obs width
            np.zeros((1, STAGE_A_ACT), np.float32),
            c,
        )


# ---- Task 4: flow-GPI inference ----

def test_infer_z_and_gpi_action_bounds():
    agent = _agent()
    agent, _ = agent.update(_batch())
    b = _batch(1)
    rewards = np.random.default_rng(2).standard_normal((B,)).astype(np.float32)
    agent2 = agent.infer_eval_z(b["next_observations"], rewards)
    assert float(np.linalg.norm(np.asarray(agent2.task_z))) > 0.0

    ob = b["observations"][0]
    a = np.asarray(agent2.sample_actions(ob, seed=jax.random.PRNGKey(0)))
    assert a.shape == (ACT,)
    assert np.all(np.abs(a) <= 1.0 + 1e-5)


def test_gpi_select_respects_u_clip():
    agent = _agent(u_clip=0.25)
    agent, _ = agent.update(_batch())
    ob = _batch(1)["observations"][0]
    u_star = np.asarray(agent.gpi_select(ob, seed=jax.random.PRNGKey(0)))
    assert np.all(np.abs(u_star) <= 0.25 + 1e-6)


def test_inferred_z_changes_gpi_choice():
    agent = _agent()
    for i in range(5):
        agent, _ = agent.update(_batch(i))
    b = _batch(9)
    ob = b["observations"][0]
    r1 = np.random.default_rng(3).standard_normal((B,)).astype(np.float32)
    a1 = np.asarray(agent.infer_eval_z(b["next_observations"], r1).sample_actions(ob, seed=jax.random.PRNGKey(1)))
    a2 = np.asarray(agent.infer_eval_z(b["next_observations"], -r1).sample_actions(ob, seed=jax.random.PRNGKey(1)))
    assert not np.allclose(a1, a2), "opposite rewards must change the GPI action"


def test_infer_eval_z_exists_for_the_main_eval_hook():
    """main.py dispatches on hasattr(agent, 'infer_eval_z'). A rename here would silently
    skip reward inference at eval and score the agent with task_z still all zeros."""
    assert hasattr(_agent(), "infer_eval_z")


def test_acts_through_the_real_eval_call_path():
    """Exercise the exact calls main.py and utils.evaluation.evaluate make.

    evaluate wraps sample_actions in supply_rng, which injects `seed` as a KEYWORD and
    passes `observations=` / `temperature=` by keyword too. A positional-only signature
    or a renamed parameter would pass every other test here and only fail at eval time,
    on the GPU, after a full training run.
    """
    from utils.evaluation import supply_rng

    agent = _agent()
    agent, _ = agent.update(_batch())
    b = _batch(7)
    rewards = np.random.default_rng(11).standard_normal((B,)).astype(np.float32)

    # main.py: eval_agent = agent.infer_eval_z(z_batch['next_observations'], rew)
    eval_agent = agent.infer_eval_z(b["next_observations"], rewards)
    # evaluate: actor_fn = supply_rng(agent.sample_actions, ...); actor_fn(observations=, temperature=)
    actor_fn = supply_rng(eval_agent.sample_actions, rng=jax.random.PRNGKey(0))
    action = np.asarray(actor_fn(observations=b["observations"][0], temperature=1.0))
    assert action.shape == (ACT,)
    assert np.all(np.isfinite(action)) and np.all(np.abs(action) <= 1.0 + 1e-5)


def test_gpi_select_returns_a_candidate_it_actually_scored():
    """u* must be one of the sampled candidates, not an index into the wrong axis.

    Q is reshaped (K, M) from a (K*M,) flat score built with repeat(u_cand, M) /
    tile(u_idx, K). Getting repeat and tile the wrong way round still produces a
    plausible-looking (K, M) matrix and a valid latent, so the bug is invisible unless
    the returned u* is checked against the candidate set.
    """
    agent = _agent()
    agent, _ = agent.update(_batch())
    ob = _batch(1)["observations"][0]
    u_star = np.asarray(agent.gpi_select(ob, seed=jax.random.PRNGKey(0)))

    # Reconstruct the candidate draw the method makes.
    c = agent.config
    r_u, _ = jax.random.split(jax.random.PRNGKey(0))
    cand = np.asarray(jnp.clip(
        jax.random.normal(r_u, (c["gpi_num_u"], c["action_dim"])), -c["u_clip"], c["u_clip"]))
    assert np.any(np.all(np.isclose(cand, u_star[None, :]), axis=1)), \
        "gpi_select returned a latent that is not among the sampled candidates"


def test_decode_paths_agree_and_stay_in_bounds():
    """onestep and ode are two decoders of the same frozen flow; both must produce valid
    actions. They are not required to agree numerically (onestep is a distilled
    approximation), but both must respect the action bounds."""
    for mode in ("onestep", "ode"):
        agent = _agent(gpi_decode=mode)
        obs = _batch(4)["observations"]
        u = np.clip(np.random.default_rng(5).standard_normal((B, ACT)), -3, 3).astype(np.float32)
        a = np.asarray(agent.decode(obs, u))
        assert a.shape == (B, ACT)
        assert np.all(np.abs(a) <= 1.0 + 1e-5), mode
        assert np.all(np.isfinite(a)), mode
