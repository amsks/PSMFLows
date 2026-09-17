"""The reference PSM critic (proto stage + reward-conditioned SF head) on latent inputs.

`agent.proto.enabled=true` (2026-09-14) adds PSM's proto successor-measure stage to
`agents/psmflow.py`: a second psi tower `proto_psi(s, z_bin, u)` indexed by a binary code,
whose TD target continues a FIXED pseudo-random latent policy keyed on (dataset row, code),
trains the basis phi together with `proto_psi`; the existing psi becomes the SF head
`psi(s, w, u)` fitted on that basis with phi stop-gradded and its target read at the ONLINE
phi the proto stage just stepped (reference `_update_sf`). Everything else -- frozen flow,
point preimages in the action slot, `infer_z`, the ddpg latent actor, task_w mixing, P=2
with `targets_uncertainty` -- is untouched. What is pinned here:

  - `utils/psm_proto.py` is the reference's code layout and seed arithmetic, and the proto
    latent is a pure function of (row, code) at a key that is NOT the run seed;
  - the SF stage sends no gradient to phi, and its target is computed at the phi the proto
    stage produced, not at `target_phi` and not at the pre-proto phi;
  - every psi action-slot input is a latent -- no decode happens inside either loss;
  - `proto.enabled=false` (the default) leaves the agent bit-for-bit where it was: the
    params after 3 updates equal a baseline captured at the commit before this feature;
  - `infer_z` is the same closed form.

Run module-per-process: `pytest tests/test_psmflow_psm_ref.py`.
"""
import math
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.psmflow import PSMFlowAgent
from tests.test_psmflow_agent import ACT, LEGACY_ARM, B, _agent, _batch
from utils.psm_common import contrastive_loss, off_diagonal_mask, ortho_loss, targets_uncertainty
from utils.psm_proto import proto_latents, proto_max_seed, proto_seed_ints, sample_z_bin

#: Small code width so a 32-row batch exercises the whole seed range.
PROTO_CFG = {"enabled": True, "max_log_seed": 8, "proto_seed": 0, "lr": 1.0e-4, "ortho_coef": None}

#: Params captured by scratchpad/capture_baseline.py at d51de4c (the commit before the
#: proto stage existed): 3 updates of `_agent()` and `_agent(**LEGACY_ARM)` on `_batch(i)`.
BASELINE = os.environ.get(
    "PSMFLOWS_PROTO_BASELINE",
    "/tmp/claude-10025/-mnt-home-amohan-git-Austin-PSMFLows/36439cd9-b8dd-4c02-82fa-75b1929f1c00/"
    "scratchpad/baseline_head_d51de4c.pkl")
BASELINE_FIELDS = ["phi", "psi", "target_phi", "target_psi", "actor", "actor_vf",
                   "sac_actor", "log_alpha", "q_dist"]


def _proto_batch(seed=0):
    """`_batch` plus the dataset ROW index the proto policy is keyed on."""
    b = _batch(seed)
    rng = np.random.default_rng(seed + 991)
    b["index"] = rng.integers(0, 1_000_000, size=B).astype(np.int64)
    return b


def _proto_agent(**overrides):
    """The first arm: the Section 10 free head + ddpg latent actor, critic swapped for PSM's."""
    kw = {**LEGACY_ARM, "proto": dict(PROTO_CFG)}
    kw.update(overrides)
    return _agent(**kw)


def _leaves(*trees):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(trees)]


def _moved(before, after):
    return any(not np.array_equal(b, a) for b, a in zip(before, after))


# --------------------------------------------------------------------------- utils/psm_proto.py
def test_z_bin_is_the_lsb_first_unpack_of_a_uniform_code():
    key = jax.random.PRNGKey(5)
    w = 8
    z = sample_z_bin(key, B, w)
    assert z.shape == (B, w) and z.dtype == jnp.float32
    assert set(np.unique(np.asarray(z))) <= {0.0, 1.0}
    codes = np.asarray(jax.random.randint(key, (B,), 0, 2 ** w))
    for k in range(w):
        np.testing.assert_array_equal(np.asarray(z[:, k]), (codes >> k) & 1)


def test_seed_arithmetic_is_the_reference_formula():
    w = 8
    z = sample_z_bin(jax.random.PRNGKey(1), B, w)
    row = np.arange(B) * 7919
    s = np.asarray(proto_seed_ints(z, row, w))
    powers = (2 ** np.arange(w))[::-1]
    expect = ((np.asarray(z).astype(np.int64) * powers).sum(1) + row) % (2 ** w + 20000)
    np.testing.assert_array_equal(s, expect)
    assert proto_max_seed(16) == 2 ** 16 + 20000


def test_proto_latent_is_a_pure_function_of_row_and_code():
    w, u_clip = 8, 1.5
    key = jax.random.PRNGKey(0)
    z = sample_z_bin(jax.random.PRNGKey(2), B, w)
    row = np.arange(B)
    u1 = np.asarray(proto_latents(proto_seed_ints(z, row, w), ACT, u_clip, key))
    u2 = np.asarray(proto_latents(proto_seed_ints(z, row, w), ACT, u_clip, key))
    np.testing.assert_array_equal(u1, u2)                      # stable across calls
    assert u1.shape == (B, ACT) and np.all(np.abs(u1) <= u_clip)
    # A different row at the same code, or a different code at the same row, is a
    # different policy draw; the same (row, code) pair anywhere in the batch is the same.
    u_row = np.asarray(proto_latents(proto_seed_ints(z, row + 1, w), ACT, u_clip, key))
    assert np.all(np.any(u_row != u1, axis=-1))
    z_flip = z.at[:, 0].set(1.0 - z[:, 0])
    u_code = np.asarray(proto_latents(proto_seed_ints(z_flip, row, w), ACT, u_clip, key))
    assert np.all(np.any(u_code != u1, axis=-1))
    z_dup = jnp.concatenate([z[:1], z[:1]]), np.array([3, 3])
    u_dup = np.asarray(proto_latents(proto_seed_ints(*z_dup, w), ACT, u_clip, key))
    np.testing.assert_array_equal(u_dup[0], u_dup[1])
    # The family is keyed on the proto seed, not on anything drawn from the run rng.
    u_other = np.asarray(proto_latents(proto_seed_ints(z, row, w), ACT, u_clip, jax.random.PRNGKey(1)))
    assert np.any(u_other != u1)


# --------------------------------------------------------------------------- construction
def test_proto_requires_the_task_vector_index_and_a_trained_actor():
    with pytest.raises(AssertionError, match="policy_index"):
        _agent(psi_form="affine", policy_index="latent", proto=dict(PROTO_CFG))
    with pytest.raises(AssertionError, match="train_actor"):
        _agent(psi_form="free", policy_index="task_vector", train_actor=False, acting="gpi",
               proto=dict(PROTO_CFG))


def test_proto_stage_needs_the_dataset_row_index():
    agent = _proto_agent()
    with pytest.raises(KeyError, match="index"):
        agent.sample_step_inputs(_batch(), jax.random.PRNGKey(0))


def test_sampled_proto_inputs_have_the_reference_layout():
    agent = _proto_agent()
    batch = _proto_batch()
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(0))
    w = agent.config["proto"]["max_log_seed"]
    assert sampled.z_bin.shape == (B, w)
    assert set(np.unique(np.asarray(sampled.z_bin))) <= {0.0, 1.0}
    assert sampled.u_proto.shape == (B, ACT)
    assert np.all(np.abs(np.asarray(sampled.u_proto)) <= agent.config["u_clip"])
    # u_proto is exactly the module's pure function of (row, z_bin) at PRNGKey(proto_seed).
    expect = proto_latents(proto_seed_ints(sampled.z_bin, batch["index"], w), ACT,
                           agent.config["u_clip"], jax.random.PRNGKey(agent.config["proto"]["proto_seed"]))
    np.testing.assert_array_equal(np.asarray(sampled.u_proto), np.asarray(expect))
    # Redrawing with another rng changes the code but not the (row, code) -> latent map.
    s2 = agent.sample_step_inputs(batch, jax.random.PRNGKey(9))
    same = np.all(np.asarray(s2.z_bin) == np.asarray(sampled.z_bin), axis=-1)
    if same.any():
        np.testing.assert_array_equal(np.asarray(s2.u_proto[same]), np.asarray(sampled.u_proto[same]))
    # The proto tower's index slot is max_log_seed wide: it accepts z_bin with u in the action slot.
    out = agent.proto_psi(batch["observations"], sampled.z_bin, sampled.u_data)
    assert out.shape == (agent.config["num_parallel"], B, agent.config["z_dim"])


# --------------------------------------------------------------------------- losses
def _proto_reference(agent, batch, sampled, phi_p, proto_p):
    """Reference `proto_loss` (archive/agents/psm.py) on latent inputs, hand-rolled."""
    c = agent.config
    obs, nxt = batch["observations"], batch["next_observations"]
    off, off_sum = off_diagonal_mask(B)
    phi_next = agent.phi(nxt, params=phi_p)
    M = agent.proto_psi(obs, sampled.z_bin, sampled.u_data, params=proto_p) @ phi_next.T
    tphi = agent.phi(nxt, params=agent.target_phi)
    tM = agent.proto_psi(nxt, sampled.z_bin, sampled.u_proto, params=agent.target_proto_psi) @ tphi.T
    mean, unc = targets_uncertainty(tM, c["num_parallel"])
    target = jax.lax.stop_gradient(mean - c["pessimism_penalty"] * unc)
    cl, _, _ = contrastive_loss(M, target, c["discount"], off, off_sum)
    ol, _, _ = ortho_loss(phi_next, off, off_sum)
    coef = c["proto"]["ortho_coef"]
    coef = c["ortho_coef"] if coef is None else coef
    return cl + coef * ol


def _sf_reference(agent, batch, sampled, phi_p, psi_p, target_phi_p):
    """Reference `sf_loss`: contrastive only, phi frozen, target measure at `target_phi_p`."""
    c = agent.config
    obs, nxt = batch["observations"], batch["next_observations"]
    off, off_sum = off_diagonal_mask(B)
    phi_next = jax.lax.stop_gradient(agent.phi(nxt, params=phi_p))
    M = agent.psi(obs, sampled.task_w, sampled.u_data, params=psi_p) @ phi_next.T
    tphi = agent.phi(nxt, params=target_phi_p)
    tM = agent.psi(nxt, sampled.task_w, sampled.u_next, params=agent.target_psi) @ tphi.T
    mean, unc = targets_uncertainty(tM, c["num_parallel"])
    target = jax.lax.stop_gradient(mean - c["pessimism_penalty"] * unc)
    cl, _, _ = contrastive_loss(M, target, c["discount"], off, off_sum)
    return cl


def test_proto_loss_matches_the_reference_and_uses_the_agents_ortho_coef():
    agent = _proto_agent()
    batch = _proto_batch()
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(0))
    loss, info = agent.proto_measure_loss(batch, sampled, agent.phi.params, agent.proto_psi.params)
    expect = _proto_reference(agent, batch, sampled, agent.phi.params, agent.proto_psi.params)
    np.testing.assert_allclose(float(loss), float(expect), rtol=1e-5)
    assert agent.config["proto"]["ortho_coef"] is None
    assert float(info["proto_ortho_weight"]) == float(agent.config["ortho_coef"])
    for k in ("proto_loss", "proto_orth_loss"):
        assert math.isfinite(float(info[k]))
    # Both phi and proto_psi receive gradient in this stage.
    g_phi, g_proto = jax.grad(lambda p, q: agent.proto_measure_loss(batch, sampled, p, q)[0],
                              argnums=(0, 1))(agent.phi.params, agent.proto_psi.params)
    assert any(np.abs(x).max() > 0 for x in _leaves(g_phi))
    assert any(np.abs(x).max() > 0 for x in _leaves(g_proto))


def test_proto_ortho_coef_override_is_honoured():
    agent = _proto_agent(proto=dict(PROTO_CFG, ortho_coef=7.0))
    batch = _proto_batch()
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(0))
    _, info = agent.proto_measure_loss(batch, sampled, agent.phi.params, agent.proto_psi.params)
    assert float(info["proto_ortho_weight"]) == 7.0


def test_sf_stage_sends_no_gradient_to_phi():
    agent = _proto_agent()
    batch = _proto_batch()
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(0))
    g_phi = jax.grad(lambda p: agent.measure_loss(batch, sampled, p, agent.psi.params)[0])(agent.phi.params)
    assert all(np.abs(x).max() == 0 for x in _leaves(g_phi))
    g_psi = jax.grad(lambda q: agent.measure_loss(batch, sampled, agent.phi.params, q)[0])(agent.psi.params)
    assert any(np.abs(x).max() > 0 for x in _leaves(g_psi))


def test_sf_target_is_read_at_the_online_phi_it_is_given():
    """Reference `_update_sf`: the target measure uses the ONLINE phi, not target_phi."""
    agent = _proto_agent()
    batch = _proto_batch()
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(0))
    # Perturb phi so online and target phi are visibly different objects.
    phi_p = jax.tree_util.tree_map(lambda x: x + 0.05, agent.phi.params)
    loss, _ = agent.measure_loss(batch, sampled, phi_p, agent.psi.params)
    at_online = _sf_reference(agent, batch, sampled, phi_p, agent.psi.params, phi_p)
    at_target = _sf_reference(agent, batch, sampled, phi_p, agent.psi.params, agent.target_phi)
    np.testing.assert_allclose(float(loss), float(at_online), rtol=1e-5)
    assert not np.isclose(float(loss), float(at_target), rtol=1e-3)


def test_apply_update_steps_sf_against_the_post_proto_phi():
    """proto step -> SF step reads phi AFTER the proto step -> actor step, replayed by hand."""
    agent = _proto_agent()
    batch = _proto_batch()
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(0))
    new, info = agent.apply_update(batch, sampled)

    (_, _), (g_phi, g_proto) = jax.value_and_grad(agent.proto_measure_loss, argnums=(2, 3), has_aux=True)(
        batch, sampled, agent.phi.params, agent.proto_psi.params)
    phi_post = agent.phi.apply_gradients(grads=g_phi)
    proto_post = agent.proto_psi.apply_gradients(grads=g_proto)
    (_, _), g_psi = jax.value_and_grad(agent.measure_loss, argnums=3, has_aux=True)(
        batch, sampled, phi_post.params, agent.psi.params)
    psi_post = agent.psi.apply_gradients(grads=g_psi)
    (_, _), g_psi_pre = jax.value_and_grad(agent.measure_loss, argnums=3, has_aux=True)(
        batch, sampled, agent.phi.params, agent.psi.params)
    psi_pre = agent.psi.apply_gradients(grads=g_psi_pre)

    for a, b in zip(_leaves(new.phi.params), _leaves(phi_post.params)):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(_leaves(new.proto_psi.params), _leaves(proto_post.params)):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(_leaves(new.psi.params), _leaves(psi_post.params)):
        np.testing.assert_array_equal(a, b)
    assert _moved(_leaves(new.psi.params), _leaves(psi_pre.params))
    # Targets: polyak on all three, and the actor stepped.
    assert _moved(_leaves(agent.target_phi), _leaves(new.target_phi))
    assert _moved(_leaves(agent.target_proto_psi), _leaves(new.target_proto_psi))
    assert _moved(_leaves(agent.target_psi), _leaves(new.target_psi))
    assert _moved(_leaves(agent.actor.params), _leaves(new.actor.params))
    for k in ("proto_loss", "proto_orth_loss", "psm_loss", "actor_loss"):
        assert math.isfinite(float(info[k])), k


def test_no_decode_inside_either_loss(monkeypatch):
    """Every psi action-slot input is a latent: the frozen flow is never called in a loss."""
    agent = _proto_agent()
    batch = _proto_batch()
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(0))

    def _boom(self, *a, **k):
        raise AssertionError("decode called inside a measure loss")

    monkeypatch.setattr(PSMFlowAgent, "decode", _boom)
    agent.proto_measure_loss(batch, sampled, agent.phi.params, agent.proto_psi.params)
    agent.measure_loss(batch, sampled, agent.phi.params, agent.psi.params)
    assert sampled.u_data.shape == (B, ACT) and sampled.u_proto.shape == (B, ACT)
    assert sampled.u_next.shape == (B, ACT)


# --------------------------------------------------------------------------- default path
def test_disabled_proto_never_steps_the_proto_tower():
    agent = _agent(**LEGACY_ARM)
    assert agent.config["proto"]["enabled"] is False
    before = _leaves(agent.proto_psi.params, agent.target_proto_psi)
    for i in range(3):
        agent, info = agent.update(_batch(i))
    assert not _moved(before, _leaves(agent.proto_psi.params, agent.target_proto_psi))
    assert not any(k.startswith("proto_") for k in info)


@pytest.mark.skipif(not os.path.exists(BASELINE), reason=f"no pre-change baseline at {BASELINE}")
@pytest.mark.parametrize("arm", ["default", "actor"])
def test_disabled_proto_is_bit_for_bit_the_pre_change_agent(arm):
    """3 updates on a random-init agent equal the params captured at d51de4c."""
    with open(BASELINE, "rb") as fh:
        base = pickle.load(fh)[arm]
    agent = _agent(**(LEGACY_ARM if arm == "actor" else {}))
    for i in range(3):
        agent, info = agent.update(_batch(i))
    for field in BASELINE_FIELDS:
        obj = getattr(agent, field)
        params = obj.params if hasattr(obj, "params") else obj
        got = _leaves(params)
        assert len(got) == len(base[field]), field
        for a, b in zip(got, base[field]):
            np.testing.assert_array_equal(a, b, err_msg=field)
    np.testing.assert_array_equal(np.asarray(agent.rng), base["rng"])
    # Every key the baseline logged is still logged with the same value. Telemetry keys
    # added since (all-arm CSV schema, e.g. psm_scalar_* on 2026-09-15) are allowed on top;
    # the params and the rng above are what pins the agent bit for bit.
    assert set(base["last_info"]) <= set(info), set(base["last_info"]) - set(info)
    for k, v in base["last_info"].items():
        assert float(info[k]) == v, k


# --------------------------------------------------------------------------- smoke + inference
def test_five_step_smoke_is_finite_and_moves_every_head():
    agent = _proto_agent()
    before = _leaves(agent.phi.params, agent.proto_psi.params, agent.psi.params, agent.actor.params)
    for i in range(5):
        agent, info = agent.update(_proto_batch(i))
        for k in ("proto_loss", "proto_orth_loss", "psm_loss", "orth_loss", "actor_loss"):
            assert math.isfinite(float(info[k])), (i, k, info[k])
    after = _leaves(agent.phi.params, agent.proto_psi.params, agent.psi.params, agent.actor.params)
    assert _moved(before[:8], after[:8])          # phi
    assert _moved(before, after)


def test_infer_z_is_unchanged_and_acting_decodes_the_actor_latent():
    agent = _proto_agent()
    for i in range(2):
        agent, _ = agent.update(_proto_batch(i))
    batch = _proto_batch(7)
    r = jnp.asarray(np.random.default_rng(3).standard_normal(B).astype(np.float32))
    z = agent.infer_z(batch["next_observations"], r)
    phi = agent.phi(batch["next_observations"])
    expect = (r.reshape(1, -1) @ phi).reshape(-1) / phi.shape[0]
    expect = jnp.sqrt(phi.shape[-1]) * expect / (jnp.linalg.norm(expect) + 1e-12)
    np.testing.assert_allclose(np.asarray(z), np.asarray(expect), rtol=1e-5, atol=1e-6)
    agent = agent.infer_eval_z(batch["next_observations"], r)
    a = agent.sample_actions(batch["observations"][0], seed=jax.random.PRNGKey(0))
    assert a.shape == (ACT,) and np.all(np.abs(np.asarray(a)) <= 1.0)
