"""`ortho_mode` and `psi_bound`: the two stabilisers of the measure loss (2026-09-07).

`docs/design/2026-09-07-antmaze-g995-collapse.md` measured the failure they answer. On
antmaze at `discount=0.995`, on three seeds agreeing to three significant figures, the
contrastive term `psm_loss` grows one decade per 31k steps, crosses
`ortho_coef * |orth_loss|` = 6.4e4 at 115-125k, and phi then collapses toward rank one
(`orth_offdiag` -> its 8192 maximum); 500-episode success falls 0.52@100k -> 0.01@300k.
`|Q|` runs 7 -> 2e8, and since phi is projected to the sphere and w(u') is unit-norm,
`A(s,u)` and `beta(s,u)` are the only free magnitudes in the model.

What is pinned here:
  - the DEFAULT (`ortho_mode='fixed'`, `psi_bound='none'`) is the PUBLISHED loss, checked
    against a from-scratch re-implementation of `sm + ortho_coef * ortho` at unbounded psi,
    and byte-identical to naming the off values explicitly -- no eval500 number moves
    because these switches exist;
  - `ortho_mode='relative'` weights ortho by `ortho_coef + ortho_rel_coef*|psm_loss|`,
    which is exactly the logged `ortho_weight`, and -- because ortho does not depend on psi
    and the weight is stop-gradded -- leaves PSI's gradient bit-identical while changing
    phi's;
  - `psi_bound='tanh'` is a bounded REPARAMETERISATION: every psi read, in the loss, in the
    actor panel and on the acting path, comes back inside +-scale, and at a scale far above
    the operating range it is the identity to numerical tolerance;
  - `psi_bound='clip_target'` bounds the TD BOOTSTRAP and leaves the forward head
    unbounded, at the Cauchy-Schwarz image `scale * z_dim` of the same psi ball;
  - the affine fast path in `_psi_q_over_indices` is skipped under the tanh bound (an
    elementwise squash on the z_dim output cannot be pushed through a w-space
    contraction), and the fallback agrees with a naive per-index computation;
  - the telemetry (`ortho_weight`, `ortho_term_abs`, `psi_absmax`, `td_target_absmean`,
    `psi_bound_frac`) is emitted in EVERY arm, so the fixed and stabilised runs share a
    CSV schema and the psm-vs-ortho crossing is readable straight off the log;
  - illegal mode names and non-positive scales are refused at `create`, not silently.

Run module-per-process: `pytest tests/test_psmflow_stabilisers.py`.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.psmflow import (
    ORTHO_MODES,
    PSI_BOUND_MODES,
    STABILITY_DEFAULTS,
    fill_stability_defaults,
)
from tests.test_psmflow_agent import ACT, OBS, _agent, _batch
from utils.psm_common import contrastive_loss, off_diagonal_mask, ortho_loss, targets_uncertainty

TELEMETRY = ("ortho_weight", "ortho_term_abs", "psi_absmean", "psi_absmax",
             "td_target_absmean", "psi_bound_frac")


def _stab(**overrides):
    """The paper-strict affine arm, which is what the antmaze collapse was measured on."""
    kw = {"psi_form": "affine", "policy_index": "latent",
          "train_actor": False, "acting": "gpi"}
    kw.update(overrides)
    return _agent(**kw)


def _leaves(agent):
    return jax.tree_util.tree_leaves((agent.phi.params, agent.psi.params))


# ---------------------------------------------------------------- 1. the default is the published loss


def test_defaults_are_off():
    a = _stab()
    for k, v in STABILITY_DEFAULTS.items():
        assert a.config[k] == v, (k, a.config[k], v)
    assert a.config["ortho_mode"] == "fixed" and a.config["psi_bound"] == "none"


def test_default_measure_loss_is_the_published_expression():
    """From scratch: loss = sm + ortho_coef * ortho at an UNBOUNDED psi and an UNCLIPPED
    target. If either switch leaked into the default path this assertion fails."""
    agent = _stab()
    batch = _batch()
    rng = jax.random.PRNGKey(0)
    sampled = agent.sample_step_inputs(batch, rng)
    loss, info = agent.measure_loss(batch, sampled, agent.phi.params, agent.psi.params)

    c = agent.config
    obs, next_obs = batch["observations"], batch["next_observations"]
    off, off_sum = off_diagonal_mask(obs.shape[0])
    index = sampled.u_index
    phi_next = agent.phi(next_obs, params=agent.phi.params)
    M = agent.psi(obs, index, sampled.u_data) @ phi_next.T
    target_phi_next = agent.phi(next_obs, params=agent.target_phi)
    M_boot = agent.psi(next_obs, index, sampled.u_next,
                       params=agent.target_psi) @ target_phi_next.T
    M_mean, M_unc = targets_uncertainty(M_boot, c["num_parallel"])
    target_M = M_mean - c["pessimism_penalty"] * M_unc
    sm, _, _ = contrastive_loss(M, jax.lax.stop_gradient(target_M), c["discount"], off, off_sum)
    ortho, _, _ = ortho_loss(phi_next, off, off_sum)

    np.testing.assert_allclose(float(loss), float(sm + c["ortho_coef"] * ortho), rtol=1e-6)
    np.testing.assert_allclose(float(info["psm_loss"]), float(sm), rtol=1e-6)
    assert float(info["ortho_weight"]) == float(c["ortho_coef"])
    assert float(info["psi_bound_frac"]) == 0.0


def test_naming_the_off_values_is_byte_identical_to_the_default():
    a2, ai = _stab().update(_batch())
    b2, bi = _stab(ortho_mode="fixed", psi_bound="none",
                   ortho_rel_coef=7.0, psi_bound_scale=3.0).update(_batch())
    for x, y in zip(_leaves(a2), _leaves(b2)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    for k in ("psm_loss", "orth_loss", "orth_offdiag"):
        assert float(ai[k]) == float(bi[k]), k


def test_the_telemetry_is_emitted_in_every_arm():
    """Same CSV schema for fixed and stabilised runs; the off values are exact constants."""
    for kw in ({}, {"ortho_mode": "relative"}, {"psi_bound": "tanh"},
               {"psi_bound": "clip_target"},
               {"ortho_mode": "relative", "psi_bound": "tanh"}):
        _, info = _stab(**kw).update(_batch())
        for k in TELEMETRY:
            assert k in info, (kw, k)
            assert math.isfinite(float(info[k])), (kw, k, info[k])


# ---------------------------------------------------------------- 2. ortho_mode=relative


def test_relative_ortho_weight_is_the_formula_and_is_larger():
    rel = 2.5
    agent = _stab(ortho_mode="relative", ortho_rel_coef=rel)
    _, info = agent.update(_batch())
    want = agent.config["ortho_coef"] + rel * abs(float(info["psm_loss"]))
    np.testing.assert_allclose(float(info["ortho_weight"]), want, rtol=1e-5)
    assert float(info["ortho_weight"]) > agent.config["ortho_coef"]
    # The whole point: the ortho term is never outgrown by the TD term.
    assert float(info["ortho_term_abs"]) > abs(float(info["psm_loss"]))


def test_relative_ortho_changes_phi_but_not_psi():
    """ortho depends on phi alone and the weight is stop-gradded, so psi's gradient -- and
    therefore psi's step -- must be bit-identical to the fixed arm."""
    batch = _batch()
    fixed = _stab()
    rel = _stab(ortho_mode="relative", ortho_rel_coef=1.0)
    f2, _ = fixed.update(batch)
    r2, _ = rel.update(batch)
    for x, y in zip(jax.tree_util.tree_leaves(f2.psi.params),
                    jax.tree_util.tree_leaves(r2.psi.params)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    assert not all(bool(jnp.array_equal(x, y)) for x, y in
                   zip(jax.tree_util.tree_leaves(f2.phi.params),
                       jax.tree_util.tree_leaves(r2.phi.params))), "phi did not move"


def test_relative_ortho_can_never_be_outgrown():
    """The invariant the fixed weight lacks.

    With w = ortho_coef + r*|sm| the ratio of the two loss terms is
    |ortho| * (ortho_coef/|sm| + r), which is bounded BELOW by r*|ortho| however far the TD
    term diverges. Under the fixed weight the same ratio is ortho_coef*|ortho|/|sm| and
    falls through 1 at |sm| = ortho_coef*|ortho| = 6.4e4 -- the measured 115-125k crossing,
    after which phi is free to buy TD loss with basis collapse.
    """
    r = 1.0
    for discount in (0.9, 0.98, 0.99):
        _, info = _stab(ortho_mode="relative", ortho_rel_coef=r,
                        discount=discount).update(_batch())
        ratio = float(info["ortho_term_abs"]) / abs(float(info["psm_loss"]))
        floor = r * abs(float(info["orth_loss"]))
        assert ratio >= floor * (1 - 1e-5), (discount, ratio, floor)
        # And the weight really does track the TD term, rather than sitting on the floor.
        assert float(info["ortho_weight"]) > _stab().config["ortho_coef"]


# ---------------------------------------------------------------- 3. psi_bound=tanh


@pytest.mark.parametrize("scale", [0.05, 1.0])
def test_tanh_bound_caps_every_psi_read(scale):
    agent = _stab(psi_bound="tanh", psi_bound_scale=scale)
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((8, OBS)).astype(np.float32)
    u = rng.standard_normal((8, ACT)).astype(np.float32)
    idx = rng.standard_normal((8, ACT)).astype(np.float32)
    bounded = np.asarray(agent.psi_b(obs, idx, u))
    raw = np.asarray(agent.psi(obs, idx, u))
    assert np.all(np.abs(bounded) <= scale + 1e-6)
    np.testing.assert_allclose(bounded, scale * np.tanh(raw / scale), rtol=1e-5, atol=1e-6)


def test_tanh_bound_at_a_huge_scale_is_the_identity():
    """S >> |psi| must reproduce the unbounded arm to numerical tolerance -- the bound is a
    ceiling, not a reshaping of the operating range."""
    batch = _batch()
    a2, ai = _stab().update(batch)
    b2, bi = _stab(psi_bound="tanh", psi_bound_scale=1.0e8).update(batch)
    np.testing.assert_allclose(float(ai["psm_loss"]), float(bi["psm_loss"]), rtol=1e-4)
    # fp32, not exact: tanh(x/1e8)*1e8 loses ~1e-7 relative on psi, which Adam's normalised
    # step can amplify on a parameter that sits near zero. atol dominates.
    for x, y in zip(_leaves(a2), _leaves(b2)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-2, atol=1e-4)
    assert float(bi["psi_bound_frac"]) == 0.0


def test_tanh_bound_reports_its_saturation_and_leaves_the_target_bounded():
    _, info = _stab(psi_bound="tanh", psi_bound_scale=1.0e-3).update(_batch())
    assert float(info["psi_bound_frac"]) > 0.5, "a scale below the operating range must bind"
    # |target_M| <= scale * ||phi|| * sqrt(z) = scale * z_dim.
    z = _stab().config["z_dim"]
    assert float(info["td_target_absmean"]) <= 1.0e-3 * z + 1e-6


def test_tanh_bound_holds_on_the_acting_path():
    agent = _stab(psi_bound="tanh", psi_bound_scale=0.01, gpi_num_u=8)
    a = agent.sample_actions(_batch()["observations"][0], seed=jax.random.PRNGKey(0))
    assert a.shape == (ACT,) and np.all(np.abs(np.asarray(a)) <= 1.0 + 1e-6)


def test_the_affine_fast_path_is_skipped_under_the_bound_and_agrees_with_it():
    """`_psi_q_over_indices` contracts in w-space when it can; the elementwise squash on the
    z_dim output cannot be pushed through that, so the bounded arm must take the generic
    path -- and the generic path must equal the naive per-index computation."""
    agent = _stab(psi_bound="tanh", psi_bound_scale=0.5)
    rng = np.random.default_rng(1)
    n, K = 6, 4
    obs = rng.standard_normal((n, OBS)).astype(np.float32)
    u = rng.standard_normal((n, ACT)).astype(np.float32)
    w = rng.standard_normal((n, agent.config["z_dim"])).astype(np.float32)
    u_index = rng.standard_normal((K, n, ACT)).astype(np.float32)
    got = np.asarray(agent._psi_q_over_indices(obs, u, w, u_index))       # (P, K, n)
    want = np.stack([np.asarray((agent.psi_b(obs, u_index[k], u) * w).sum(-1))
                     for k in range(K)], axis=1)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    assert np.all(np.abs(got) <= 0.5 * agent.config["z_dim"] * np.abs(w).max() * 8)


# ---------------------------------------------------------------- 4. psi_bound=clip_target


def test_clip_target_bounds_the_bootstrap_and_not_the_head():
    scale = 1.0e-4
    agent = _stab(psi_bound="clip_target", psi_bound_scale=scale)
    _, info = agent.update(_batch())
    cap = scale * agent.config["z_dim"]
    assert float(info["td_target_absmean"]) <= cap + 1e-9
    assert float(info["psi_bound_frac"]) > 0.5, "a cap below the operating range must bind"
    # The FORWARD head is untouched: psi still reads its unbounded magnitude.
    _, none_info = _stab().update(_batch())
    np.testing.assert_allclose(float(info["psi_absmax"]), float(none_info["psi_absmax"]),
                               rtol=1e-6)


def test_clip_target_at_a_huge_scale_is_the_identity():
    batch = _batch()
    a2, ai = _stab().update(batch)
    b2, bi = _stab(psi_bound="clip_target", psi_bound_scale=1.0e9).update(batch)
    for x, y in zip(_leaves(a2), _leaves(b2)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    assert float(ai["psm_loss"]) == float(bi["psm_loss"])
    assert float(bi["psi_bound_frac"]) == 0.0


# ---------------------------------------------------------------- 5. arms, guards, backfill


@pytest.mark.parametrize("ortho_mode", ORTHO_MODES)
@pytest.mark.parametrize("psi_bound", PSI_BOUND_MODES)
def test_every_combination_takes_a_finite_step(ortho_mode, psi_bound):
    agent = _stab(ortho_mode=ortho_mode, psi_bound=psi_bound)
    before = _leaves(agent)
    agent2, info = agent.update(_batch())
    for k in ("psm_loss", "orth_loss", "psi_q_spread_rel", "w_enc_spread", *TELEMETRY):
        assert math.isfinite(float(info[k])), (ortho_mode, psi_bound, k, info[k])
    assert not all(bool(jnp.array_equal(x, y)) for x, y in zip(before, _leaves(agent2)))


def test_the_free_head_also_honours_the_bound():
    """psi_form=free has no factorisation to skip, but the ceiling is on psi, not on the
    parameterisation, so the ablation arm must be bounded too."""
    agent = _agent(psi_form="free", policy_index="latent", train_actor=False, acting="gpi",
                   psi_bound="tanh", psi_bound_scale=0.02)
    rng = np.random.default_rng(2)
    out = np.asarray(agent.psi_b(rng.standard_normal((4, OBS)).astype(np.float32),
                                 rng.standard_normal((4, ACT)).astype(np.float32),
                                 rng.standard_normal((4, ACT)).astype(np.float32)))
    assert np.all(np.abs(out) <= 0.02 + 1e-6)


def test_illegal_modes_are_refused():
    with pytest.raises(AssertionError, match="ortho_mode"):
        _stab(ortho_mode="proportional")
    with pytest.raises(AssertionError, match="psi_bound"):
        _stab(psi_bound="clip")
    with pytest.raises(AssertionError, match="psi_bound_scale"):
        _stab(psi_bound="tanh", psi_bound_scale=0.0)
    with pytest.raises(AssertionError, match="ortho_rel_coef"):
        _stab(ortho_mode="relative", ortho_rel_coef=-1.0)


def test_a_config_predating_the_keys_reads_as_off():
    """The `fill_actor_defaults` contract, for these keys: a flags.json written before they
    existed must restore onto the OFF behaviour rather than raising."""
    import ml_collections

    from agents.psmflow import PSMFlowAgent, get_config

    cfg = get_config()
    with cfg.unlocked():
        for k in STABILITY_DEFAULTS:
            del cfg[k]
        cfg["allow_untrained_flow"] = True
        cfg["z_dim"] = 16
    agent = PSMFlowAgent.create(0, np.zeros((1, OBS), np.float32),
                                np.zeros((1, ACT), np.float32), cfg)
    for k, v in STABILITY_DEFAULTS.items():
        assert agent.config[k] == v, k
    fill_stability_defaults(ml_collections.ConfigDict({"agent_name": "fql"}))
    fill_stability_defaults({})
