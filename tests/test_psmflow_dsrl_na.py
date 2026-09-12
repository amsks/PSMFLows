"""DSRL-NA's dual critic: `dsrl_na.enabled` (2026-09-08).

The 2026-09-08 audit found that what this repo shipped as "DSRL-NA" was DSRL's ACTOR with
none of DSRL's CRITIC (`docs/design/2026-09-08-critic-signal-and-dsrl-na.md` 1). Real
DSRL-NA is a dual critic:

    qa(s, a)   scalar TD on the task's REAL reward, bootstrap a' = G(s', pi(s'))
    qw(s, u)   regressed onto qa(s, G(s, u)) at prior latents, `inner_steps` per update
               -- and the ONLY thing the latent actor climbs.

The arm is deliberately NOT zero-shot; it is the upper bound on whether the frozen flow can
be steered at all. What is pinned here:

  - the DEFAULT path is untouched: with `dsrl_na.enabled=False` the two heads never step
    and `_actor_q` still reads psi;
  - qa's TD target is exactly `r + gamma * mask * min_e qa_bar(s', G(s', u'))`, computed
    against a hand-rolled reference -- this is the one place the agent reads a reward, so a
    silent sign or mask error would turn the upper bound into noise;
  - qw's regression target is qa at the DECODE of the latent, and the fit improves over the
    inner steps;
  - the actor's Q is qw and nothing else, and its task slot is ZEROED (DSRL-NA is
    single-task; a task-conditioned policy would condition on a vector its critic ignores);
  - the frozen flow and the ddpg actor head are untouched;
  - `create` refuses every half-configured arm, which is the failure this whole file exists
    to prevent recurring.

Run module-per-process: `pytest tests/test_psmflow_dsrl_na.py`.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_psmflow_agent import ACT, OBS, _agent, _batch

#: The affine substrate the arm sits on. DSRL-NA does not touch psi, so unlike the
#: dsrl_sac-on-psi arm it composes with the paper-strict defaults.
AFFINE = {"psi_form": "affine", "policy_index": "latent"}

#: Small enough to build fast; every structural property under test is width-independent.
NA_CFG = dict(enabled=True, discount=0.99, tau=0.005, lr=3.0e-4,
              hidden_dim=32, hidden_layers=2, layer_norm=True,
              num_ensembles=2, inner_steps=3, n_latent=1)


def _actor_cfg(**kw):
    """The actor sub-dict DSRL-NA needs (ConfigDict wants the whole thing)."""
    base = dict(hidden_dim=64, hidden_layers=2, embedding_layers=2,
                vf_hidden_dim=64, vf_hidden_layers=2, flow_steps=4, bc_coeff=0.0,
                index_panel=0, q_coeff=1.0, na_coeff=1.0, na_candidates=4, na_states=8,
                na_advantage_weight=False, entropy="auto", target_entropy=0.0,
                init_alpha=1.0, lr_alpha=3.0e-4, log_std_min=-20.0, log_std_max=2.0,
                prior_init=True, prior_init_std=0.3, layer_norm=True)
    base.update(kw)
    return base


def _na_agent(**overrides):
    """The DSRL-NA arm: DSRL's actor and DSRL's critic together."""
    kw = {**AFFINE, "train_actor": True, "acting": "actor", "actor_mode": "dsrl_sac",
          "actor": _actor_cfg(), "dsrl_na": dict(NA_CFG), "u_clip": 1.5}
    kw.update(overrides)
    return _agent(**kw)


def _na_batch(seed=0):
    """`_batch` plus the reward and mask columns qa's TD target reads."""
    b = _batch(seed)
    rng = np.random.default_rng(seed + 77)
    b["rewards"] = rng.choice([-1.0, 0.0], size=len(b["observations"])).astype(np.float32)
    b["masks"] = rng.choice([0.0, 1.0], size=len(b["observations"])).astype(np.float32)
    return b


def _leaves(*trees):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(trees)]


def _moved(before, after):
    return any(not np.array_equal(b, a) for b, a in zip(before, after))


# --------------------------------------------------------------------------- default path
def test_default_agent_never_steps_the_na_heads():
    """`dsrl_na.enabled=False` (the default) leaves both heads exactly as initialised."""
    agent = _agent()
    assert agent.config["dsrl_na"]["enabled"] is False
    before = _leaves(agent.qa, agent.target_qa, agent.qw)
    for i in range(3):
        agent, info = agent.update(_na_batch(i))
    assert not _moved(before, _leaves(agent.qa, agent.target_qa, agent.qw))
    assert not any(k.startswith("na_") for k in info), [k for k in info if k.startswith("na_")]


def test_default_actor_q_still_reads_psi():
    """The dispatch in `_actor_q` is additive: without the flag it is the psi readout."""
    agent = _agent(**AFFINE, train_actor=True, acting="actor", actor=_actor_cfg(bc_coeff=1.0))
    batch = _na_batch()
    rng = jax.random.PRNGKey(0)
    sampled = agent.sample_step_inputs(batch, rng)
    u_a = jnp.zeros((len(batch["observations"]), ACT), jnp.float32)
    Q, _ = agent._actor_q(batch["observations"], u_a, sampled)
    psi_q = (agent.psi_b(batch["observations"], agent._index(sampled), u_a)
             * sampled.task_w).sum(-1)
    # P=2, actor_pessimism_penalty=0.5 => mean - spread/2 == min over the ensemble
    np.testing.assert_allclose(np.asarray(Q), np.asarray(psi_q.min(0)), rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- qa: real-reward TD
def test_qa_target_is_the_real_reward_td_target():
    """r + gamma * mask * min_e qa_bar(s', G(s', u')), against a hand-rolled reference."""
    agent = _na_agent()
    batch = _na_batch()
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(3))
    loss, info = agent.dsrl_qa_loss(batch, sampled, agent.qa.params)

    na = agent.config["dsrl_na"]
    u_next = agent._deploy_latent(batch["next_observations"],
                                  jnp.zeros_like(sampled.task_w), sampled.flow_noise)
    a_next = agent.decode(batch["next_observations"], u_next)
    q_next = agent.qa(batch["next_observations"], a_next, params=agent.target_qa).min(0)
    ref_target = batch["rewards"] + na["discount"] * batch["masks"] * q_next
    q = agent.qa(batch["observations"], batch["actions"], params=agent.qa.params)
    ref_loss = jnp.square(q - ref_target[None]).mean()

    np.testing.assert_allclose(float(loss), float(ref_loss), rtol=1e-5)
    np.testing.assert_allclose(float(info["na_qa_target"]), float(ref_target.mean()), rtol=1e-5)


def test_qa_loss_reads_the_reward():
    """Flipping every reward moves the loss -- the arm is reward-specific by construction."""
    agent = _na_agent()
    batch = _na_batch()
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(3))
    base, _ = agent.dsrl_qa_loss(batch, sampled, agent.qa.params)
    shifted = {**batch, "rewards": batch["rewards"] - 1.0}
    other, _ = agent.dsrl_qa_loss(shifted, sampled, agent.qa.params)
    assert abs(float(base) - float(other)) > 1e-6


def test_qa_bootstrap_carries_no_gradient_to_the_actor():
    """qa is a critic, not a path into the policy: the bootstrap action is stop-gradded.

    Without that stop_gradient the actor would receive a gradient that MINIMISES the TD
    error -- pushing the policy toward whatever action makes qa easiest to fit, which is
    not policy improvement.
    """
    agent = _na_agent()
    batch = _na_batch()
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(3))

    def through_the_actor(sac_params):
        a = agent.replace(sac_actor=agent.sac_actor.replace(params=sac_params))
        return a.dsrl_qa_loss(batch, sampled, agent.qa.params)[0]

    g = jax.grad(through_the_actor)(agent.sac_actor.params)
    assert all(float(np.abs(np.asarray(x)).sum()) == 0.0
               for x in jax.tree_util.tree_leaves(g))


# --------------------------------------------------------------------------- qw: distillation
def test_qw_regresses_onto_qa_at_the_decode():
    """The target is qa(s, G(s, u)), not qa(s, a_data) and not psi."""
    agent = _na_agent()
    batch = _na_batch()
    key = jax.random.PRNGKey(11)
    loss, info = agent.dsrl_qw_loss(batch, agent.qw.params, key)

    c = agent.config
    n, B = c["dsrl_na"]["n_latent"], len(batch["observations"])
    u = jnp.clip(jax.random.normal(key, (n, B, ACT)), -c["u_clip"], c["u_clip"])
    obs_r = jnp.broadcast_to(batch["observations"][None], (n, B, OBS)).reshape(n * B, OBS)
    u_r = u.reshape(n * B, ACT)
    ref_target = agent.qa(obs_r, agent.decode(obs_r, u_r)).min(0)
    ref = jnp.square(agent.qw(obs_r, u_r, params=agent.qw.params) - ref_target).mean()

    np.testing.assert_allclose(float(loss), float(ref), rtol=1e-5)
    np.testing.assert_allclose(float(info["na_qw_target"]), float(ref_target.mean()), rtol=1e-5)


def test_qw_fit_improves_over_the_inner_steps():
    """`inner_steps` gradient steps on a frozen qa reduce the regression loss."""
    agent = _na_agent(dsrl_na={**NA_CFG, "inner_steps": 1, "lr": 1e-2})
    batch = _na_batch()
    key = jax.random.PRNGKey(5)
    qw, first, last = agent.qw, None, None
    for i in range(12):
        (loss, _), g = jax.value_and_grad(agent.dsrl_qw_loss, argnums=1, has_aux=True)(
            batch, qw.params, key)
        qw = qw.apply_gradients(grads=g)
        first = float(loss) if i == 0 else first
        last = float(loss)
    assert last < first, (first, last)


# --------------------------------------------------------------------------- the arm
def test_update_steps_the_na_heads_and_the_actor_only():
    agent = _na_agent()
    frozen = _leaves(agent.flow_vf, agent.flow_onestep, agent.actor, agent.psi_a)
    before = _leaves(agent.qa, agent.qw, agent.sac_actor, agent.log_alpha)
    for i in range(2):
        agent, info = agent.update(_na_batch(i))
    assert _moved(before, _leaves(agent.qa, agent.qw, agent.sac_actor, agent.log_alpha))
    assert not _moved(frozen, _leaves(agent.flow_vf, agent.flow_onestep,
                                      agent.actor, agent.psi_a))
    for k in ("na_qa_loss", "na_qw_loss", "na_qw_spread_over_u",
              "na_signal_over_disagreement", "actor_q", "actor_alpha"):
        assert math.isfinite(float(info[k])), (k, info[k])


def test_actor_q_is_qw_and_the_task_slot_is_zeroed():
    agent = _na_agent()
    batch = _na_batch()
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(1))
    u_a = jnp.zeros((len(batch["observations"]), ACT), jnp.float32)
    Q, q_ens = agent._actor_q(batch["observations"], u_a, sampled)
    np.testing.assert_allclose(np.asarray(Q), np.asarray(agent.qw(batch["observations"], u_a)),
                               rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(Q), np.asarray(q_ens))
    np.testing.assert_array_equal(np.asarray(agent._actor_w(sampled.task_w)),
                                  np.zeros_like(np.asarray(sampled.task_w)))


def test_acting_ignores_the_inferred_task_vector():
    """Single-task by construction: task_z must not reach the deployed policy."""
    agent = _na_agent()
    obs = jnp.asarray(_na_batch()["observations"][0])
    a0 = agent.replace(task_z=jnp.zeros_like(agent.task_z)).sample_actions(
        obs, seed=jax.random.PRNGKey(0), temperature=0)
    a1 = agent.replace(task_z=jnp.ones_like(agent.task_z)).sample_actions(
        obs, seed=jax.random.PRNGKey(0), temperature=0)
    np.testing.assert_allclose(np.asarray(a0), np.asarray(a1), rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- the guards
@pytest.mark.parametrize("overrides, needle", [
    ({"actor_mode": "ddpg", "train_actor": True, "acting": "actor"}, "dsrl_sac"),
    ({"acting": "gpi"}, "deploys the latent actor"),
    ({"actor": _actor_cfg(bc_coeff=1.0)}, "bc_coeff"),
    ({"actor": _actor_cfg(q_coeff=0.0)}, "q_coeff"),
])
def test_create_refuses_a_half_configured_arm(overrides, needle):
    with pytest.raises(AssertionError, match=needle):
        _na_agent(**overrides)


# ------------------------------------------------ Item 3 Arm B: measure_u_samples
# Lives here rather than in its own file because it shares `_na_batch`'s habit of adding
# the columns `_batch` omits; the seam itself has nothing to do with DSRL-NA.
def _mixture_batch(seed=0, k=1, scale=0.3):
    """`_batch` plus the stored EM preimage mixture columns the extra draws read."""
    b = _batch(seed)
    n = len(b["observations"])
    rng = np.random.default_rng(seed + 5)
    b["noise_preimage_mean"] = np.repeat(b["noise_preimage"][:, None], k, 1).astype(np.float32)
    b["noise_preimage_cov"] = np.broadcast_to(
        (scale ** 2) * np.eye(ACT, dtype=np.float32), (n, k, ACT, ACT)).copy()
    w = rng.random((n, k)).astype(np.float32)
    b["noise_preimage_weights"] = w / w.sum(1, keepdims=True)
    return b


def test_measure_u_samples_defaults_to_the_published_loss():
    """1 (the default) leaves `u_extra` unset and the loss bit-identical."""
    agent = _agent()
    assert agent.config["measure_u_samples"] == 1
    batch = _mixture_batch()
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(0))
    assert sampled.u_extra is None
    loss, info = agent.measure_loss(batch, sampled, agent.phi.params, agent.psi.params)
    assert math.isfinite(float(loss))
    assert float(info["u_extra_dist"]) == 0.0


def test_extra_latents_come_from_the_stored_mixture():
    """Draws sit around the stored mean at the stored width, inside the u box."""
    agent = _agent(measure_u_samples=5, measure_u_mixture_shrink=1.0)
    batch = _mixture_batch(scale=0.25)
    u = np.asarray(agent._sample_preimage_mixture(batch, 4, jax.random.PRNGKey(1)))
    assert u.shape == (4, len(batch["observations"]), ACT)
    assert np.all(np.abs(u) <= agent.config["u_clip"] + 1e-6)
    # mean over the four draws is within a few standard errors of the stored mean
    err = np.abs(u.mean(0) - batch["noise_preimage_mean"][:, 0]).mean()
    assert err < 0.25 * 3 / math.sqrt(4), err
    assert 0.15 < u.std(0).mean() < 0.35, u.std(0).mean()


def test_extra_latents_enter_the_measure_loss():
    """The loss moves, `u_extra_dist` reports the augmentation width, and it still trains."""
    agent = _agent(measure_u_samples=4, measure_u_mixture_shrink=1.0)
    batch = _mixture_batch(scale=0.4)
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(2))
    assert sampled.u_extra.shape == (3, len(batch["observations"]), ACT)
    loss, info = agent.measure_loss(batch, sampled, agent.phi.params, agent.psi.params)
    assert math.isfinite(float(loss))
    assert float(info["u_extra_dist"]) > 0.0
    assert 0.0 <= float(info["u_extra_clipfrac"]) <= 1.0

    one = _agent()
    s1 = one.sample_step_inputs(batch, jax.random.PRNGKey(2))
    l1, _ = one.measure_loss(batch, s1, one.phi.params, one.psi.params)
    assert abs(float(loss) - float(l1)) > 1e-8

    before = _leaves(agent.psi, agent.phi)
    agent, upd = agent.update(batch)
    assert _moved(before, _leaves(agent.psi, agent.phi))
    assert math.isfinite(float(upd["psm_loss"]))


def test_zero_width_mixture_reproduces_the_single_u_loss():
    """With the posterior collapsed onto the point inverse the extra rows are duplicates."""
    batch = _mixture_batch(scale=0.0)
    many = _agent(measure_u_samples=4, measure_u_mixture_shrink=1.0)
    sm = many.sample_step_inputs(batch, jax.random.PRNGKey(4))
    lm, im = many.measure_loss(batch, sm, many.phi.params, many.psi.params)
    one = _agent()
    s1 = one.sample_step_inputs(batch, jax.random.PRNGKey(4))
    l1, _ = one.measure_loss(batch, s1, one.phi.params, one.psi.params)
    np.testing.assert_allclose(float(lm), float(l1), rtol=1e-5)
    np.testing.assert_allclose(float(im["u_extra_dist"]), 0.0, atol=1e-5)


def test_jitter_source_stays_near_u_data_and_scales_with_sigma():
    """`measure_u_source=jitter`: a ball of the configured width around the point inverse.

    The width is not a free parameter -- `tools/diag_mixture_decode.py` picks it from the
    decode-error ladder (sigma 0.3 decodes at 0.103 against the point inverse's own p90 of
    0.128 on cube), which is what makes these latents preimages of the SAME action where
    the stored posterior's samples are not (0.205, 60% of the way to a prior draw).
    """
    batch = _mixture_batch()
    prev = 0.0
    for sig in (0.1, 0.3, 1.0):
        agent = _agent(measure_u_samples=4, measure_u_source="jitter",
                       measure_u_jitter_std=sig)
        sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(9))
        _, info = agent.measure_loss(batch, sampled, agent.phi.params, agent.psi.params)
        d = float(info["u_extra_dist"])
        # E||sigma * eps|| over d_a dims, loose so the test is not a chi distribution table
        assert 0.5 * sig * math.sqrt(ACT) < d < 1.6 * sig * math.sqrt(ACT), (sig, d)
        assert d > prev
        prev = d


def test_jitter_source_needs_no_mixture_columns():
    """It reads only u_data, so it runs on the npz every published number used."""
    agent = _agent(measure_u_samples=3, measure_u_source="jitter")
    batch = _batch()          # no noise_preimage_mean/cov/weights at all
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(2))
    assert sampled.u_extra.shape == (2, len(batch["observations"]), ACT)
    agent, info = agent.update(batch)
    assert math.isfinite(float(info["psm_loss"]))


def test_create_refuses_a_zero_width_jitter():
    with pytest.raises(AssertionError, match="measure_u_jitter_std"):
        _agent(measure_u_samples=4, measure_u_source="jitter", measure_u_jitter_std=0.0)


# ------------------------------------------------ Item 3 Arm C: measure_u_mixture_shrink
def test_create_refuses_the_mixture_without_an_explicit_shrink():
    """The one knob with no safe default: cube wants 0.5, antmaze 1.0, pointmaze 0.5."""
    with pytest.raises(AssertionError, match="measure_u_mixture_shrink"):
        _agent(measure_u_samples=4)          # measure_u_source defaults to "mixture"


def test_shrink_zero_collapses_onto_the_component_means():
    """c = 0 is the posterior's mean, NOT u_data -- the floor the sweep flattens out at."""
    batch = _mixture_batch(scale=0.6)
    agent = _agent(measure_u_samples=4, measure_u_mixture_shrink=0.0)
    u = np.asarray(agent._sample_preimage_mixture(batch, 3, jax.random.PRNGKey(3), 0.0))
    mean0 = np.clip(batch["noise_preimage_mean"][:, 0], -agent.config["u_clip"],
                    agent.config["u_clip"])
    for k in range(3):
        np.testing.assert_allclose(u[k], mean0, rtol=1e-5, atol=1e-5)


def test_shrink_scales_the_spread_but_not_the_mean():
    """cov -> c^2 cov: the draw's std scales with c while its centre does not move.

    Centre means the component mean CLIPPED TO THE BOX, and that is not a technicality: the
    stored posterior mean lies outside [-u_clip, u_clip] on a real fraction of rows (it sits
    2.79 from the point inverse on pointmaze), so shrinking collapses onto the clipped mean
    and the box, not the posterior, decides where the extra latents end up there.
    """
    batch = _mixture_batch(scale=0.5)
    agent = _agent(measure_u_samples=4, measure_u_mixture_shrink=1.0)
    n, box = 256, agent.config["u_clip"]
    wide = np.asarray(agent._sample_preimage_mixture(batch, n, jax.random.PRNGKey(7), 1.0))
    tight = np.asarray(agent._sample_preimage_mixture(batch, n, jax.random.PRNGKey(7), 0.25))
    mean0 = np.clip(batch["noise_preimage_mean"][:, 0], -box, box)
    # Tolerances are ~4 standard errors of the sample mean at each scale (per-dim sd 0.5),
    # loosened by 1.5 for the rows the clip touches.
    assert np.abs(wide.mean(0) - mean0).max() < 4 * 0.5 / math.sqrt(n) * 1.5
    assert np.abs(tight.mean(0) - mean0).max() < 4 * 0.5 / math.sqrt(n) * 1.5
    ratio = tight.std(0).mean() / wide.std(0).mean()
    assert 0.2 < ratio < 0.3, ratio


def test_shrink_reaches_the_loss_and_is_recorded():
    """A smaller c pulls the extra latents in, and `u_extra_dist` reports it."""
    batch = _mixture_batch(scale=0.6)
    dists = []
    for c in (0.2, 1.0):
        agent = _agent(measure_u_samples=4, measure_u_mixture_shrink=c)
        sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(11))
        _, info = agent.measure_loss(batch, sampled, agent.phi.params, agent.psi.params)
        dists.append(float(info["u_extra_dist"]))
    assert dists[0] < dists[1], dists


def test_clipfrac_reports_the_box_wall():
    """A posterior sitting outside the box shows up as a large `u_extra_clipfrac`.

    Not decoration: on pointmaze the stored mean is 2.79 from the point inverse and a real
    fraction of rows have it outside [-u_clip, u_clip], so where the draws land is decided by
    the box rather than by the posterior. That has to be visible in the run metrics.
    """
    inside = _mixture_batch(scale=0.2)
    outside = {**inside,
               "noise_preimage_mean": inside["noise_preimage_mean"] + 20.0}
    fracs = []
    for batch in (inside, outside):
        agent = _agent(measure_u_samples=4, measure_u_mixture_shrink=1.0)
        sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(13))
        _, info = agent.measure_loss(batch, sampled, agent.phi.params, agent.psi.params)
        fracs.append(float(info["u_extra_clipfrac"]))
    assert fracs[0] < 0.05, fracs
    assert fracs[1] > 0.95, fracs


# ------------------------------------------------ Arm D1: dsrl_na.reward_source
def test_reward_source_defaults_to_the_real_reward():
    agent = _na_agent()
    assert agent.config["dsrl_na"]["reward_source"] == "real"
    batch = _na_batch()
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(3))
    _, info = agent.dsrl_qa_loss(batch, sampled, agent.qa.params)
    assert not any(k.startswith("na_rhat") for k in info)


def test_phi_readout_replaces_the_reward_with_its_linear_reconstruction():
    """r_hat = phi(s')^T project(E[(r+shift) phi]) -- the deployed zero-shot channel."""
    from utils.psm_common import project_z
    agent = _na_agent(dsrl_na={**NA_CFG, "reward_source": "phi_readout"})
    batch = _na_batch()
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(3))
    loss, info = agent.dsrl_qa_loss(batch, sampled, agent.qa.params)

    na, c = agent.config["dsrl_na"], agent.config
    r_shift = batch["rewards"] + na["reward_shift"]
    ph = agent.phi(batch["next_observations"])
    w = project_z((r_shift.reshape(1, -1) @ ph).reshape(-1) / ph.shape[0], c["norm_z"])
    r_hat = ph @ w
    q_next = agent.qa(batch["next_observations"],
                      agent.decode(batch["next_observations"],
                                   agent._deploy_latent(batch["next_observations"],
                                                        jnp.zeros_like(sampled.task_w),
                                                        sampled.flow_noise)),
                      params=agent.target_qa).min(0)
    ref_t = r_hat + na["discount"] * batch["masks"] * q_next
    q = agent.qa(batch["observations"], batch["actions"], params=agent.qa.params)
    np.testing.assert_allclose(float(loss), float(jnp.square(q - ref_t[None]).mean()), rtol=1e-5)
    np.testing.assert_allclose(float(info["na_rhat_mean"]), float(r_hat.mean()), rtol=1e-5)
    assert math.isfinite(float(info["na_rhat_corr"]))


def test_phi_readout_still_trains_and_is_not_the_real_reward():
    agent = _na_agent(dsrl_na={**NA_CFG, "reward_source": "phi_readout"})
    batch = _na_batch()
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(3))
    real, _ = _na_agent().dsrl_qa_loss(batch, sampled, agent.qa.params)
    readout, _ = agent.dsrl_qa_loss(batch, sampled, agent.qa.params)
    assert abs(float(real) - float(readout)) > 1e-6
    before = _leaves(agent.qa, agent.qw, agent.sac_actor)
    agent, info = agent.update(batch)
    assert _moved(before, _leaves(agent.qa, agent.qw, agent.sac_actor))
    for k in ("na_qa_loss", "na_rhat_corr", "na_rhat_std"):
        assert math.isfinite(float(info[k])), (k, info[k])


def test_create_refuses_an_unknown_reward_source():
    with pytest.raises(AssertionError, match="reward_source"):
        _na_agent(dsrl_na={**NA_CFG, "reward_source": "made_up"})


# ------------------------------------------------ whitened reward inference
def test_reward_inference_defaults_to_the_closed_form():
    agent = _agent()
    assert agent.config["reward_inference"] == "closed_form"
    b = _na_batch()
    z = agent.infer_z(b["next_observations"], b["rewards"])
    ph = agent.phi(b["next_observations"])
    from utils.psm_common import project_z
    ref = project_z((b["rewards"].reshape(1, -1) @ ph).reshape(-1) / ph.shape[0],
                    agent.config["norm_z"])
    np.testing.assert_allclose(np.asarray(z), np.asarray(ref), rtol=1e-5, atol=1e-6)


def test_whitened_inference_solves_the_normal_equations():
    agent = _agent(reward_inference="whitened", reward_inference_eps=1e-6)
    b = _na_batch()
    ph = np.asarray(agent.phi(b["next_observations"]), np.float64)
    r = np.asarray(b["rewards"], np.float64)
    gram = ph.T @ ph / len(ph)
    ref = np.linalg.solve(gram + 1e-6 * np.eye(gram.shape[0]), ph.T @ r / len(ph))
    ref = ref / np.linalg.norm(ref) * math.sqrt(len(ref))
    got = np.asarray(agent.infer_z(b["next_observations"], b["rewards"]), np.float64)
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-4)


def test_whitening_is_a_no_op_when_the_gram_is_the_identity():
    """The claim that makes this safe on cube: identical wherever E[phi phi^T] = I."""
    agent = _agent(reward_inference="whitened", reward_inference_eps=0.0)
    b = _na_batch()
    ph = np.asarray(agent.phi(b["next_observations"]), np.float64)
    gram = ph.T @ ph / len(ph)
    # The test agent's phi is NOT orthonormal, so verify the identity claim directly on a
    # synthetic orthonormal basis rather than pretending this one is.
    d, n = 8, 4096
    rng = np.random.default_rng(0)
    q = np.linalg.qr(rng.standard_normal((n, d)))[0] * math.sqrt(n)   # columns orthonormal
    r = rng.standard_normal(n)
    cf = q.T @ r / n
    wh = np.linalg.solve(q.T @ q / n, q.T @ r / n)
    np.testing.assert_allclose(cf, wh, rtol=1e-8, atol=1e-10)
    assert np.linalg.norm(gram - np.eye(gram.shape[0])) > 0.0     # and this one is not


def test_create_refuses_an_unknown_inference_mode():
    with pytest.raises(AssertionError, match="reward_inference"):
        _agent(reward_inference="magic")


# ------------------------------------------------ Arm D2: task-conditioned, zero-shot
def _d2_agent(**overrides):
    """Arm D2: critics and actor take w, and the reward IS phi(s')^T w."""
    na = {**NA_CFG, "reward_source": "synthetic_w", "task_conditioned": True}
    return _na_agent(dsrl_na=na, **overrides)


def test_d2_reward_is_the_synthetic_task_and_never_the_real_one():
    """No real reward enters training -- that is what makes the arm zero-shot."""
    agent = _d2_agent()
    batch = _na_batch()
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(3))
    loss, info = agent.dsrl_qa_loss(batch, sampled, agent.qa.params)

    w = sampled.task_w
    r_w = (agent.phi(batch["next_observations"]) * w).sum(-1)
    na = agent.config["dsrl_na"]
    u_n = agent._deploy_latent(batch["next_observations"], w, sampled.flow_noise)
    a_n = agent.decode(batch["next_observations"], u_n)
    q_n = agent.qa(batch["next_observations"], agent._na_in(a_n, w),
                   params=agent.target_qa).min(0)
    ref_t = r_w + na["discount"] * batch["masks"] * q_n
    q = agent.qa(batch["observations"], agent._na_in(batch["actions"], w),
                 params=agent.qa.params)
    np.testing.assert_allclose(float(loss), float(jnp.square(q - ref_t[None]).mean()), rtol=1e-5)

    # Flipping the real reward must not move the loss at all.
    flipped = {**batch, "rewards": -batch["rewards"] - 5.0}
    l2, _ = agent.dsrl_qa_loss(flipped, sampled, agent.qa.params)
    np.testing.assert_allclose(float(loss), float(l2), rtol=1e-6)
    assert math.isfinite(float(info["na_rw_corr_to_real"]))


def test_d2_conditions_every_head_on_the_task_vector():
    agent = _d2_agent()
    batch = _na_batch()
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(1))
    # the task slot reaches the actor rather than being zeroed
    np.testing.assert_allclose(np.asarray(agent._actor_w(sampled.task_w)),
                               np.asarray(sampled.task_w), rtol=1e-6)
    u_a = jnp.zeros((len(batch["observations"]), ACT), jnp.float32)
    Q, _ = agent._actor_q(batch["observations"], u_a, sampled)
    ref = agent.qw(batch["observations"], agent._na_in(u_a, sampled.task_w))
    np.testing.assert_allclose(np.asarray(Q), np.asarray(ref), rtol=1e-6)
    # and a different w gives a different Q, i.e. w is actually read
    other = jnp.roll(sampled.task_w, 1, axis=0)
    assert not np.allclose(np.asarray(ref),
                           np.asarray(agent.qw(batch["observations"],
                                               agent._na_in(u_a, other))))


def test_d2_trains_and_stays_finite():
    agent = _d2_agent()
    before = _leaves(agent.qa, agent.qw, agent.sac_actor)
    for i in range(2):
        agent, info = agent.update(_na_batch(i))
    assert _moved(before, _leaves(agent.qa, agent.qw, agent.sac_actor))
    for k in ("na_qa_loss", "na_qw_loss", "na_rw_std", "actor_q"):
        assert math.isfinite(float(info[k])), (k, info[k])


def test_task_vector_report_separates_a_training_draw_from_an_unrelated_one():
    """The membership check: a real phi(s') direction scores far above the random null."""
    agent = _d2_agent()
    b = _na_batch()
    nxt = b["next_observations"]
    from utils.psm_common import project_z
    inside = project_z(agent.phi(nxt)[0], agent.config["norm_z"])
    r_in = agent.task_vector_report(nxt, inside, jax.random.PRNGKey(0))
    assert float(r_in["w_cos_phi_max"]) > 0.99            # it IS one of them
    assert float(r_in["w_cos_phi_max_over_null"]) > 3.0

    rng = np.random.default_rng(0)
    out = jnp.asarray(rng.standard_normal(agent.config["z_dim"]), jnp.float32)
    r_out = agent.task_vector_report(nxt, out, jax.random.PRNGKey(0))
    assert float(r_out["w_cos_phi_max"]) < float(r_in["w_cos_phi_max"])


@pytest.mark.parametrize("na, needle", [
    ({**NA_CFG, "reward_source": "synthetic_w"}, "task_conditioned=true"),
    ({**NA_CFG, "task_conditioned": True}, "synthetic_w"),
])
def test_create_refuses_a_half_configured_d2(na, needle):
    with pytest.raises(AssertionError, match=needle):
        _na_agent(dsrl_na=na)


# --------------------------------------------------------------- Arm D1b: the held readout
def test_refit_na_reward_sets_w_and_matches_the_reward_scale():
    """`refit_na_reward` fits w by the closed form and rescales r_hat onto r's std.

    Arm D1's failure mode was the scale: the sphere projection fixes ||w||, which left
    r_hat's std at 13.6 against the real reward's 0.15. The rescale is what makes a critic
    trained on this channel comparable to one trained on the real reward.
    """
    agent = _na_agent(dsrl_na={**NA_CFG, "reward_source": "phi_readout_fixed"})
    b = _na_batch(0)
    before = np.asarray(agent.na_rw).copy()
    new, info = agent.refit_na_reward(b["next_observations"], b["rewards"])
    assert np.allclose(before, 0.0), "na_rw must start at its zero init"
    assert not np.allclose(np.asarray(new.na_rw), 0.0), "refit must set na_rw"

    shift = 1.0   # dsrl_na.reward_shift default; NA_CFG does not set it
    r_shift = np.asarray(b["rewards"]) + shift
    r_hat = np.asarray(new.phi(b["next_observations"]) @ new.na_rw) * float(new.na_rw_scale)
    assert np.isclose(r_hat.std(), r_shift.std(), rtol=1e-4), (
        f"rescaled r_hat std {r_hat.std()} != reward std {r_shift.std()}")
    assert set(info) >= {"na_rhat_corr_infit", "na_rw_scale", "na_reward_std"}


def test_refit_na_reward_reports_a_heldout_correlation():
    """The in-fit correlation is upward-biased; the held-out one is the number to read."""
    agent = _na_agent(dsrl_na={**NA_CFG, "reward_source": "phi_readout_fixed"})
    fit, ho = _na_batch(0), _na_batch(5)
    _, info = agent.refit_na_reward(fit["next_observations"], fit["rewards"],
                                    ho["next_observations"], ho["rewards"])
    assert "na_rhat_corr_heldout" in info
    assert -1.0 <= info["na_rhat_corr_heldout"] <= 1.0
    _, no_ho = agent.refit_na_reward(fit["next_observations"], fit["rewards"])
    assert "na_rhat_corr_heldout" not in no_ho, "held-out key only with a second batch"


def test_phi_readout_fixed_reward_is_constant_between_refits():
    """The whole point of D1b: Q_A sees ONE reward function until the next refit.

    Under `phi_readout` w is refit on each 256-row batch, so the reward a given row is
    assigned changes every step. Here two different batches must be scored by the SAME w.
    """
    agent = _na_agent(dsrl_na={**NA_CFG, "reward_source": "phi_readout_fixed"})
    fit = _na_batch(0)
    new, _ = agent.refit_na_reward(fit["next_observations"], fit["rewards"])
    w1 = np.asarray(new.na_rw).copy()

    b2 = _na_batch(9)
    r_a = np.asarray(new.phi(b2["next_observations"]) @ new.na_rw) * float(new.na_rw_scale)
    # A different batch must not move w: nothing in the loss refits it.
    sampled = new.sample_step_inputs(b2, jax.random.PRNGKey(0))
    _, info = new.dsrl_qa_loss(b2, sampled, new.qa.params)
    assert np.allclose(np.asarray(new.na_rw), w1), "the qa loss must not refit na_rw"
    r_b = np.asarray(new.phi(b2["next_observations"]) @ new.na_rw) * float(new.na_rw_scale)
    assert np.allclose(r_a, r_b)
    assert "na_rhat_corr_inbatch" in info and "na_rw_scale" in info


def test_create_refuses_phi_readout_fixed_without_a_refit_interval():
    """With no refit the held w stays at zero and the reward is identically 0."""
    with pytest.raises(AssertionError, match="reward_refit_every"):
        _na_agent(dsrl_na={**NA_CFG, "reward_source": "phi_readout_fixed",
                           "reward_refit_every": 0})
