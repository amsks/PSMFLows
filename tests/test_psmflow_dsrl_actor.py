"""DSRL-style latent actor on the affine critic: `actor_mode` and `actor.index_panel`.

Added 2026-09-06 with docs/design/2026-09-06-dsrl-actor-audit.md, which measured that the
shipped latent actor climbs `psi(s, u, u')^T w` at ONE prior index draw per batch element,
at cosine -0.49 / -0.20 against the gradient of the panel max the deployed GPI rule
maximises. Two switchable fixes, both defaulted OFF:

  actor.index_panel = K > 0   the actor's Q becomes max over K prior indices;
  actor_mode = dsrl_sac       tanh-Gaussian latent policy + SAC entropy (DSRL's actor);
  actor_mode = dsrl_na        regression onto the GPI argmax over prior draws (DSRL-NA's
                              prior-only signal, distilled into the policy rather than
                              into a second critic).

What is pinned here:
  - the DEFAULT path is untouched: `actor_mode=ddpg, index_panel=0` steps exactly the
    params it stepped before, leaves the DSRL heads inert, and `_actor_q` reduces to the
    pre-existing single-draw readout op for op;
  - the affine fast path in `_psi_q_over_indices` (A(s,u) computed once, contracted with K
    encoder outputs) agrees with K explicit psi calls -- the whole point of the affine head
    is that the index panel is nearly free, and a wrong einsum would be silent;
  - each DSRL mode trains its own head, emits finite diagnostics, and leaves the frozen
    flow and the ddpg head alone;
  - the NA regression target is always a PRIOR draw (the support device that replaces the
    BC anchor), so it lives in the u_clip box;
  - the tanh-Gaussian's log-prob matches a brute-force change-of-variables computation;
  - the create() guards refuse the combinations that would be silently mis-typed.

Run module-per-process: `pytest tests/test_psmflow_dsrl_actor.py` (a single-process run of
the whole suite hits a pre-existing XLA-CPU error in test_decode_recovery).
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_psmflow_agent import ACT, _agent, _batch
from utils.psm_common import targets_uncertainty
from utils.psm_networks import tanh_gaussian_sample

# The affine substrate the whole ablation runs on.
AFFINE = {"psi_form": "affine", "policy_index": "latent"}


def _tree_equal(a, b):
    return all(bool(jnp.array_equal(x, y))
               for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


def _actor_arm(**overrides):
    """The affine critic with a latent actor trained and deployed."""
    kw = {**AFFINE, "train_actor": True, "acting": "actor"}
    kw.update(overrides)
    return _agent(**kw)


def _actor_cfg(**kw):
    """actor sub-dict with the shipped values plus overrides (ConfigDict needs it whole)."""
    base = dict(hidden_dim=512, hidden_layers=2, embedding_layers=2,
                vf_hidden_dim=512, vf_hidden_layers=4, flow_steps=10, bc_coeff=1.0,
                index_panel=0, q_coeff=1.0, na_coeff=1.0, na_candidates=16, na_states=256,
                na_advantage_weight=False, entropy="auto", target_entropy=0.0,
                init_alpha=1.0, lr_alpha=3.0e-4, log_std_min=-20.0, log_std_max=2.0)
    base.update(kw)
    return base


# --------------------------------------------------------------------------- defaults


def test_default_config_is_the_shipped_ddpg_actor():
    agent = _agent()
    assert agent.config["actor_mode"] == "ddpg"
    assert int(agent.config["actor"]["index_panel"]) == 0


def test_ddpg_path_leaves_the_dsrl_heads_inert():
    """actor_mode=ddpg must step the NoiseConditionedActor and nothing new."""
    agent = _actor_arm()
    stepped, info = agent.update(_batch())
    assert not _tree_equal(agent.actor.params, stepped.actor.params)
    assert _tree_equal(agent.sac_actor.params, stepped.sac_actor.params), "sac head trained"
    assert _tree_equal(agent.log_alpha.params, stepped.log_alpha.params), "alpha trained"
    assert "alpha_loss" not in info and "actor_na_error" not in info


def test_index_panel_zero_is_the_pre_existing_readout():
    """`_actor_q` at K=0 == psi at the single index draw, op for op."""
    agent = _actor_arm()
    batch = _batch()
    rng = jax.random.PRNGKey(3)
    sampled = agent.sample_step_inputs(batch, rng)
    obs, w = batch["observations"], sampled.task_w
    u_a = agent.config["u_clip"] * agent.actor(obs, w, sampled.flow_noise)
    Q, Qs = agent._actor_q(obs, u_a, sampled)
    ref_Qs = (agent.psi(obs, agent._index(sampled), u_a) * w).sum(-1)
    qm, qu = targets_uncertainty(ref_Qs, agent.config["num_parallel"])
    ref_Q = qm - agent.config["actor_pessimism_penalty"] * qu
    np.testing.assert_array_equal(np.asarray(Qs), np.asarray(ref_Qs))
    np.testing.assert_array_equal(np.asarray(Q), np.asarray(ref_Q))


def test_index_panel_does_not_change_the_measure_step():
    """An actor-only knob must leave the psi/phi one-step delta byte-identical."""
    a1 = _actor_arm(actor=_actor_cfg(index_panel=0))
    a2 = _actor_arm(actor=_actor_cfg(index_panel=4))
    u1, _ = a1.update(_batch())
    u2, _ = a2.update(_batch())
    assert _tree_equal(u1.psi.params, u2.psi.params)
    assert _tree_equal(u1.phi.params, u2.phi.params)
    assert not _tree_equal(u1.actor.params, u2.actor.params), "panel changed nothing"


# ------------------------------------------------------------- affine fast path is right


@pytest.mark.parametrize("psi_form", ["affine", "free"])
def test_psi_q_over_indices_matches_explicit_psi_calls(psi_form):
    """The A-once-contract-K-times shortcut must equal K full psi evaluations."""
    agent = _agent(psi_form=psi_form, policy_index="latent",
                   train_actor=False, acting="gpi")
    batch = _batch(1)
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(5))
    obs, w, u = batch["observations"], sampled.task_w, sampled.u_data
    K, B = 3, obs.shape[0]
    u_idx = jnp.clip(jax.random.normal(jax.random.PRNGKey(9), (K, B, ACT)), -3.0, 3.0)
    got = agent._psi_q_over_indices(obs, u, w, u_idx)                     # (P, K, B)
    ref = jnp.stack([(agent.psi(obs, u_idx[k], u) * w).sum(-1) for k in range(K)], axis=1)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=2e-4, atol=2e-4)


def test_index_panel_takes_the_max_over_the_panel():
    agent = _actor_arm(actor=_actor_cfg(index_panel=5))
    batch = _batch(2)
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(7))
    obs, w = batch["observations"], sampled.task_w
    u_a = agent.config["u_clip"] * agent.actor(obs, w, sampled.flow_noise)
    Q, _ = agent._actor_q(obs, u_a, sampled)
    u_idx = jnp.clip(jax.random.normal(jax.random.fold_in(agent.rng, 112), (5, obs.shape[0], ACT)),
                     -3.0, 3.0)
    Qk = agent._psi_q_over_indices(obs, u_a, w, u_idx)
    qm, qu = targets_uncertainty(Qk, agent.config["num_parallel"])
    ref = (qm - agent.config["actor_pessimism_penalty"] * qu).max(0)
    np.testing.assert_allclose(np.asarray(Q), np.asarray(ref), rtol=1e-6, atol=1e-6)
    # A max over 5 draws can only be >= the single-draw readout it replaces on average.
    single, _ = _actor_arm(actor=_actor_cfg(index_panel=0))._actor_q(obs, u_a, sampled)
    assert float(Q.mean()) >= float(single.mean()) - 1e3  # loose: different random indices


# ------------------------------------------------------------------------- dsrl_sac arm


def test_dsrl_sac_trains_its_own_head():
    agent = _actor_arm(actor_mode="dsrl_sac", actor=_actor_cfg(index_panel=4, bc_coeff=0.0))
    stepped, info = agent.update(_batch())
    for k in ("actor_loss", "actor_q", "actor_logp", "actor_alpha", "actor_entropy",
              "alpha_loss", "log_alpha"):
        assert math.isfinite(float(info[k])), (k, info[k])
    assert not _tree_equal(agent.sac_actor.params, stepped.sac_actor.params)
    assert not _tree_equal(agent.log_alpha.params, stepped.log_alpha.params)
    assert _tree_equal(agent.actor.params, stepped.actor.params), "ddpg head trained too"
    assert _tree_equal(agent.flow_vf, stepped.flow_vf), "frozen flow moved"


def test_dsrl_sac_entropy_fixed_pins_alpha():
    agent = _actor_arm(actor_mode="dsrl_sac",
                       actor=_actor_cfg(index_panel=4, entropy="fixed", init_alpha=0.25))
    stepped, info = agent.update(_batch())
    assert _tree_equal(agent.log_alpha.params, stepped.log_alpha.params)
    assert "alpha_loss" not in info
    np.testing.assert_allclose(float(info["actor_alpha"]), 0.25, rtol=1e-5)


def test_tanh_gaussian_log_prob_matches_change_of_variables():
    mu = jnp.array([[0.3, -1.2]])
    log_std = jnp.array([[-0.5, 0.25]])
    noise = jnp.array([[1.1, -0.4]])
    scale = 3.0
    u, logp = tanh_gaussian_sample(mu, log_std, noise, scale)
    pre = mu + jnp.exp(log_std) * noise
    ref = jnp.sum(-0.5 * ((pre - mu) / jnp.exp(log_std)) ** 2 - log_std
                  - 0.5 * jnp.log(2 * jnp.pi), axis=-1)
    ref = ref - jnp.sum(jnp.log(1 - jnp.tanh(pre) ** 2 + 1e-6), axis=-1)
    np.testing.assert_allclose(np.asarray(logp), np.asarray(ref), rtol=1e-5, atol=1e-5)
    assert (np.abs(np.asarray(u)) <= scale + 1e-6).all()


def test_dsrl_acting_uses_the_gaussian_head():
    agent = _actor_arm(actor_mode="dsrl_sac", actor=_actor_cfg(index_panel=4))
    agent = agent.replace(task_z=jnp.ones_like(agent.task_z))
    obs = jnp.zeros((agent.config["ob_dims"][0],), jnp.float32)
    a = np.asarray(agent.sample_actions(obs, seed=jax.random.PRNGKey(0)))
    assert a.shape == (ACT,) and np.isfinite(a).all() and (np.abs(a) <= 1.0 + 1e-6).all()
    ddpg = _actor_arm().replace(task_z=jnp.ones_like(agent.task_z))
    assert not np.allclose(a, np.asarray(ddpg.sample_actions(obs, seed=jax.random.PRNGKey(0))))


# -------------------------------------------------------------------------- dsrl_na arm


def test_dsrl_na_trains_and_regresses_onto_a_prior_draw():
    agent = _actor_arm(actor_mode="dsrl_na",
                       actor=_actor_cfg(index_panel=4, na_candidates=8, na_states=16,
                                        q_coeff=0.0, bc_coeff=0.0))
    stepped, info = agent.update(_batch())
    for k in ("actor_loss", "actor_na_error", "actor_na_mse", "actor_na_advantage",
              "actor_na_target_norm"):
        assert math.isfinite(float(info[k])), (k, info[k])
    assert float(info["actor_na_error"]) >= 0.0
    # Without advantage weighting the optimised term IS the plain MSE.
    np.testing.assert_allclose(float(info["actor_na_error"]), float(info["actor_na_mse"]),
                               rtol=1e-6)
    # Every candidate is a clipped prior draw, so the target norm cannot exceed the box.
    assert float(info["actor_na_target_norm"]) <= 3.0 * math.sqrt(ACT) + 1e-5
    assert not _tree_equal(agent.sac_actor.params, stepped.sac_actor.params)
    assert _tree_equal(agent.actor.params, stepped.actor.params)


def test_dsrl_na_target_is_the_gpi_argmax():
    agent = _actor_arm(actor_mode="dsrl_na",
                       actor=_actor_cfg(index_panel=3, na_candidates=6, na_states=16))
    batch = _batch(4)
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(11))
    obs, w = batch["observations"][:8], sampled.task_w[:8]
    key = jax.random.PRNGKey(21)
    u_star, adv = agent._gpi_argmax_target(obs, w, key)
    assert u_star.shape == (8, ACT)
    assert (np.abs(np.asarray(u_star)) <= 3.0 + 1e-6).all()
    # Advantage is max - mean over the candidate set, hence non-negative.
    assert (np.asarray(adv) >= -1e-3).all()
    # And the chosen latent is genuinely the argmax of the panel-max score.
    k_u, k_i = jax.random.split(key)
    u_cand = jnp.clip(jax.random.normal(k_u, (6, 8, ACT)), -3.0, 3.0)
    u_idx = jnp.clip(jax.random.normal(k_i, (3, 8, ACT)), -3.0, 3.0)

    def score(u_m):
        Qk = agent._psi_q_over_indices(obs, u_m, w, u_idx)
        qm, qu = targets_uncertainty(Qk, agent.config["num_parallel"])
        return (qm - agent.config["actor_pessimism_penalty"] * qu).max(0)

    Q = jax.vmap(score)(u_cand)
    best = jnp.argmax(Q, axis=0)
    np.testing.assert_allclose(
        np.asarray(u_star),
        np.asarray(jnp.take_along_axis(u_cand, best[None, :, None], axis=0)[0]),
        rtol=1e-6, atol=1e-6)


def test_dsrl_na_mse_is_unweighted_under_advantage_weighting():
    """`actor_na_mse` must stay the plain mean so it is comparable to the independence
    floor across both settings; `actor_na_error` is the reweighted term being optimised."""
    agent = _actor_arm(actor_mode="dsrl_na",
                       actor=_actor_cfg(index_panel=3, na_candidates=6, na_states=16,
                                        na_advantage_weight=True))
    _, info = agent.update(_batch())
    assert math.isfinite(float(info["actor_na_mse"]))
    # A non-degenerate reweighting makes the two differ; equality would mean the weights
    # collapsed to 1 and the remedy is not actually being applied.
    assert float(info["actor_na_error"]) != float(info["actor_na_mse"])


def test_dsrl_na_advantage_weighting_changes_the_step():
    a1 = _actor_arm(actor_mode="dsrl_na",
                    actor=_actor_cfg(index_panel=3, na_candidates=6, na_states=16,
                                     na_advantage_weight=False))
    a2 = _actor_arm(actor_mode="dsrl_na",
                    actor=_actor_cfg(index_panel=3, na_candidates=6, na_states=16,
                                     na_advantage_weight=True))
    u1, _ = a1.update(_batch())
    u2, _ = a2.update(_batch())
    assert _tree_equal(u1.psi.params, u2.psi.params), "an actor knob moved the measure"
    assert not _tree_equal(u1.sac_actor.params, u2.sac_actor.params)


# ----------------------------------------------------------------------------- guards


def test_dsrl_mode_requires_train_actor():
    with pytest.raises(AssertionError, match="train_actor=true"):
        _agent(**AFFINE, train_actor=False, acting="gpi", actor_mode="dsrl_sac")


def test_index_panel_requires_latent_index():
    with pytest.raises(AssertionError, match="policy_index=task_vector"):
        _agent(psi_form="free", policy_index="task_vector", train_actor=True, acting="actor",
               actor=_actor_cfg(index_panel=4))


def test_index_panel_refuses_expectile():
    with pytest.raises(AssertionError, match="index_agg=max"):
        _agent(**AFFINE, train_actor=True, acting="actor", index_agg="expectile",
               actor=_actor_cfg(index_panel=4))


def test_unknown_actor_mode_is_refused():
    with pytest.raises(AssertionError, match="actor_mode"):
        _agent(**AFFINE, train_actor=True, acting="actor", actor_mode="dsrl_whatever")
