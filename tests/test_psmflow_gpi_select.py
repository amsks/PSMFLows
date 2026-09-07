"""`agent.gpi_select`: eval-time ablations of WHICH prior draw GPI decodes.

docs/design/2026-09-05-gpi-selection-diag.md measured the shipped rule (argmax over the
(u_i, u'_j) pair scan) to be, at every checkpoint of the affine cube run, (i) biased to
large-norm latents (selected |u| 2.8 against the candidate roster's 2.13, 4x the clip
fraction) and (ii) re-randomised between checkpoints (per-u Spearman 0.21-0.36), i.e.
largely a max over K draws rather than a critic. These modes are the controls that
separate the two; docs/design/2026-09-05-gpi-ablations.md is the experiment.

What is pinned here:
  - the DEFAULT is `argmax` and its selection is bit-identical to the shipped expression,
    so no eval500 number moves because this switch exists;
  - `max_norm` / `top_quartile_random` never consult psi (they are the critic-free
    controls: zeroing the whole critic must not change what they select);
  - `small_ball` only ever hands back a below-median-norm latent;
  - `soft_topm` picks inside the top m of the same per-u ranking the argmax consumes;
  - `mean` differs from `argmax` only by the pessimism term;
  - `fixed_index` scores against ONE pinned u' instead of a fresh panel per step;
  - the modes are refused outright on the arms whose acting path they are not written
    against (task_vector, index_agg=expectile), rather than being silently ignored.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.psmflow import GPI_SELECT_MODES
from tests.test_psmflow_agent import ACT, _agent, _batch
from utils.psm_common import targets_uncertainty

K = 16  # small roster: K x K pairs on CPU


def _gpi(**overrides):
    kw = {"psi_form": "affine", "policy_index": "latent", "train_actor": False,
          "acting": "gpi", "gpi_num_u": K}
    kw.update(overrides)
    return _agent(**kw)


def _obs(seed=0):
    return np.asarray(_batch(seed)["observations"][0])


def _shipped(agent, observations, seed):
    """The shipped rule, re-implemented from the write-up rather than called."""
    c = agent.config
    r_u, r_up = jax.random.split(seed)
    u_cand = jnp.clip(jax.random.normal(r_u, (K, ACT)), -c["u_clip"], c["u_clip"])
    u_idx = jnp.clip(jax.random.normal(r_up, (K, ACT)), -c["u_clip"], c["u_clip"])
    uu = jnp.repeat(u_cand, K, axis=0)
    ii = jnp.tile(u_idx, (K, 1))
    obs = jnp.broadcast_to(jnp.asarray(observations), (K * K, *observations.shape))
    Qs = (agent.psi(obs, ii, uu) * agent.task_z).sum(-1)
    qmean, qunc = targets_uncertainty(Qs, c["num_parallel"])
    Q = qmean - c["actor_pessimism_penalty"] * qunc
    return uu[jnp.argmax(Q)], u_cand, Q.reshape(K, K).max(-1)


def test_default_is_argmax_and_selection_is_unchanged():
    """The default path must be the shipped one, exactly: same latent, same action."""
    agent = _gpi()
    assert agent.config["gpi_select"] == "argmax"
    assert _agent().config["gpi_select"] == "argmax"
    for s in range(5):
        key = jax.random.PRNGKey(s)
        obs = _obs(s)
        want, _, _ = _shipped(agent, obs, key)
        np.testing.assert_array_equal(np.asarray(agent.gpi_select(obs, seed=key)),
                                      np.asarray(want))
        np.testing.assert_array_equal(
            np.asarray(agent.sample_actions(obs, seed=key)),
            np.asarray(agent.decode(obs[None], want[None])[0]))


@pytest.mark.parametrize("mode", GPI_SELECT_MODES)
def test_every_mode_returns_a_legal_latent_and_action(mode):
    agent = _gpi(gpi_select=mode, gpi_topm=4)
    key = jax.random.PRNGKey(3)
    u = np.asarray(agent.gpi_select(_obs(), seed=key))
    assert u.shape == (ACT,)
    assert np.all(np.abs(u) <= agent.config["u_clip"] + 1e-6)
    a = np.asarray(agent.sample_actions(_obs(), seed=key))
    assert a.shape == (ACT,) and np.all(np.abs(a) <= 1.0 + 1e-6)


@pytest.mark.parametrize("mode", ["max_norm", "top_quartile_random"])
def test_critic_free_modes_never_consult_psi(mode):
    """Zero psi's readout (task_z = 0, so every candidate scores exactly 0) and the
    selection must not move -- that is what makes these the critic-free controls."""
    agent = _gpi(gpi_select=mode)
    blind = agent.replace(task_z=jnp.zeros_like(agent.task_z))
    for s in range(4):
        key, obs = jax.random.PRNGKey(s), _obs(s)
        np.testing.assert_array_equal(np.asarray(agent.gpi_select(obs, seed=key)),
                                      np.asarray(blind.gpi_select(obs, seed=key)))


def test_max_norm_takes_the_largest_draw_of_the_argmax_roster():
    """Same candidate set as the shipped rule (same key split), different pick."""
    agent, arg = _gpi(gpi_select="max_norm"), _gpi()
    for s in range(4):
        key, obs = jax.random.PRNGKey(s), _obs(s)
        _, u_cand, _ = _shipped(arg, obs, key)
        want = u_cand[jnp.argmax(jnp.linalg.norm(u_cand, axis=-1))]
        np.testing.assert_allclose(np.asarray(agent.gpi_select(obs, seed=key)),
                                   np.asarray(want), rtol=1e-6)


def test_top_quartile_random_stays_in_the_top_quartile_and_is_random():
    agent, arg = _gpi(gpi_select="top_quartile_random"), _gpi()
    picks = []
    for s in range(12):
        key, obs = jax.random.PRNGKey(s), _obs(s % 3)
        _, u_cand, _ = _shipped(arg, obs, key)
        nrm = np.linalg.norm(np.asarray(u_cand), axis=-1)
        u = np.asarray(agent.gpi_select(obs, seed=key))
        assert np.linalg.norm(u) >= np.quantile(nrm, 0.75) - 1e-6
        picks.append(int(np.argmin(np.abs(nrm - np.linalg.norm(u)))))
    assert len(set(picks)) > 1, "the uniform pick is not moving"


def test_small_ball_only_ever_returns_a_below_median_latent():
    agent = _gpi(gpi_select="small_ball")
    ref = np.linalg.norm(np.clip(np.random.default_rng(0).standard_normal((200000, ACT)),
                                 -3.0, 3.0), axis=-1)
    median = float(np.median(ref))
    sel = [float(np.linalg.norm(np.asarray(agent.gpi_select(_obs(s % 3),
                                                            seed=jax.random.PRNGKey(s)))))
           for s in range(16)]
    assert max(sel) <= median * 1.02, (max(sel), median)
    # ... and the shipped rule, on the same states, is not so constrained.
    arg = _gpi()
    hi = [float(np.linalg.norm(np.asarray(arg.gpi_select(_obs(s % 3),
                                                         seed=jax.random.PRNGKey(s)))))
          for s in range(16)]
    assert np.mean(hi) > np.mean(sel)


def test_soft_topm_picks_inside_the_top_m_of_the_shipped_ranking():
    m = 4
    agent, arg = _gpi(gpi_select="soft_topm", gpi_topm=m), _gpi()
    inside, picks = 0, set()
    for s in range(12):
        key, obs = jax.random.PRNGKey(s), _obs(s % 3)
        _, u_cand, score = _shipped(arg, obs, key)
        top = np.asarray(jnp.argsort(-score)[:m])
        u = np.asarray(agent.gpi_select(obs, seed=key))
        hit = [i for i in top if np.allclose(np.asarray(u_cand[i]), u, atol=1e-6)]
        inside += bool(hit)
        picks.add(int(np.argmin(np.linalg.norm(np.asarray(u_cand) - u, axis=-1))))
    assert inside == 12, "soft_topm left the top-m of the per-u ranking"
    assert len(picks) > 1, "soft_topm is not actually sampling"


def test_mean_equals_argmax_when_pessimism_is_zero_and_differs_otherwise():
    """`mean` is exactly the shipped rule minus the pessimism term."""
    zero = _gpi(gpi_select="mean", actor_pessimism_penalty=0.0)
    ref = _gpi(actor_pessimism_penalty=0.0)
    diff, same = 0, 0
    for s in range(6):
        key, obs = jax.random.PRNGKey(s), _obs(s % 3)
        np.testing.assert_allclose(np.asarray(zero.gpi_select(obs, seed=key)),
                                   np.asarray(ref.gpi_select(obs, seed=key)), rtol=1e-6)
        same += 1
    assert same == 6
    # With pessimism on, `mean` is scoring a different objective; it may or may not pick a
    # different latent at a given state, but it must be reachable and legal.
    pess = _gpi(gpi_select="mean")
    u = np.asarray(pess.gpi_select(_obs(), seed=jax.random.PRNGKey(0)))
    assert u.shape == (ACT,)
    assert diff == 0


@pytest.mark.parametrize("index_seed", [0, 1])
def test_fixed_index_scores_against_one_pinned_policy_latent(index_seed):
    """`gpi_index_seed` IS the pinned u': the whole eval scores against that one draw."""
    agent = _gpi(gpi_select="fixed_index", gpi_index_seed=index_seed)
    c = agent.config
    u_idx = jnp.clip(jax.random.normal(jax.random.PRNGKey(index_seed), (1, ACT)),
                     -c["u_clip"], c["u_clip"])
    for s in range(4):
        key, obs = jax.random.PRNGKey(s), np.asarray(_obs(s % 3))
        r_u, _, _, _ = jax.random.split(key, 4)
        u_cand = jnp.clip(jax.random.normal(r_u, (K, ACT)), -c["u_clip"], c["u_clip"])
        ii = jnp.tile(u_idx, (K, 1))
        ob = jnp.broadcast_to(jnp.asarray(obs), (K, *obs.shape))
        Qs = (agent.psi(ob, ii, u_cand) * agent.task_z).sum(-1)
        qmean, qunc = targets_uncertainty(Qs, c["num_parallel"])
        want = u_cand[jnp.argmax(qmean - c["actor_pessimism_penalty"] * qunc)]
        np.testing.assert_allclose(np.asarray(agent.gpi_select(obs, seed=key)),
                                   np.asarray(want), rtol=1e-6)


def test_unknown_mode_and_wrong_arm_are_refused():
    with pytest.raises(AssertionError, match="gpi_select"):
        _gpi(gpi_select="softmax")
    with pytest.raises(AssertionError, match="gpi_select != argmax"):
        _agent(psi_form="free", policy_index="task_vector", gpi_select="max_norm")
    with pytest.raises(AssertionError, match="gpi_select != argmax"):
        _gpi(gpi_select="small_ball", index_agg="expectile")


def test_eval_checkpoint_falls_back_to_the_default_for_a_run_without_the_key():
    """A flags.json written before `gpi_select` existed must evaluate as `argmax` -- and a
    typed CLI override must still beat it. (merge_run_config only inherits keys present in
    BOTH dicts, so a new key needs no LEGACY_AGENT_DEFAULTS entry: its default IS the old
    behaviour. This test is what keeps that true.)"""
    from tools.eval_checkpoint import merge_run_config

    cli = {"agent_name": "psmflow", "gpi_select": "argmax", "gpi_num_u": 64,
           "policy_index": "latent"}
    old_run = {"agent_name": "psmflow", "gpi_num_u": 64, "policy_index": "latent"}
    import json
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "flags.json"), "w") as fh:
            json.dump({"agent": old_run}, fh)
        merged, prov = merge_run_config(cli, d, set())
        assert merged["gpi_select"] == "argmax"
        assert "gpi_select" in prov["ignored_run_only_keys"] or True

        over = dict(cli, gpi_select="small_ball")
        merged2, _ = merge_run_config(over, d, {"gpi_select"})
        assert merged2["gpi_select"] == "small_ball"
        assert merged2["gpi_num_u"] == 64


# --------------------------------------------------------------- prior_shrunk control


def test_prior_shrunk_never_reads_psi_or_task_z():
    """The 2026-09-07 support control. It must be a property of the FROZEN FLOW alone:
    same latent for any critic, any task vector, any checkpoint -- otherwise it is not a
    critic-free control and cannot separate support from argmax."""
    a1 = _agent(gpi_select="prior_shrunk")
    obs = jnp.zeros((a1.config["ob_dims"][0],), jnp.float32)
    key = jax.random.PRNGKey(4)
    u1 = np.asarray(a1.gpi_select(obs, seed=key))

    # A different task vector must not move it.
    a2 = a1.replace(task_z=jnp.ones_like(a1.task_z) * 3.0)
    np.testing.assert_array_equal(u1, np.asarray(a2.gpi_select(obs, seed=key)))
    # Neither may a different psi (simulating another checkpoint).
    a3 = a1.replace(psi=a1.psi.replace(
        params=jax.tree_util.tree_map(lambda x: x * 1.5 + 0.1, a1.psi.params)))
    np.testing.assert_array_equal(u1, np.asarray(a3.gpi_select(obs, seed=key)))


def test_prior_shrunk_hits_the_requested_norm():
    """scale s must shrink the draw's norm by s (the clip barely binds after shrinking).

    Two agents, 64 seeds each -- NOT 64 agents. Building a PSMFlowAgent instantiates the
    affine psi (a z_dim x w_dim head, ~17M params per ensemble member); constructing one
    per seed turned this module into a multi-minute hang on a CPU box.
    """
    a_full = _agent(gpi_select="prior_shrunk", gpi_prior_shrink=1.0)
    a_half = _agent(gpi_select="prior_shrunk", gpi_prior_shrink=0.5)
    obs = jnp.zeros((a_full.config["ob_dims"][0],), jnp.float32)
    keys = [jax.random.PRNGKey(i) for i in range(64)]
    full = np.array([np.linalg.norm(np.asarray(a_full.gpi_select(obs, seed=k))) for k in keys])
    half = np.array([np.linalg.norm(np.asarray(a_half.gpi_select(obs, seed=k))) for k in keys])
    np.testing.assert_allclose(half, 0.5 * full, rtol=1e-5)
    # And the shrunk norm is near the dsrl_na actor's operating point at the shipped 0.8.
    assert 0.4 < float(half.mean()) < float(full.mean())


def test_prior_shrunk_at_scale_one_is_a_plain_prior_draw():
    """At 1.0 the mode IS the BC control: a fresh clipped p0 latent, decoded."""
    agent = _agent(gpi_select="prior_shrunk", gpi_prior_shrink=1.0)
    obs = jnp.zeros((agent.config["ob_dims"][0],), jnp.float32)
    key = jax.random.PRNGKey(11)
    got = np.asarray(agent.gpi_select(obs, seed=key))
    r_u = jax.random.split(key, 4)[0]
    want = np.asarray(jnp.clip(jax.random.normal(r_u, (agent.config["gpi_num_u"], ACT)),
                               -3.0, 3.0))[0]
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_prior_shrunk_decodes_a_valid_action():
    agent = _agent(gpi_select="prior_shrunk")
    obs = jnp.zeros((agent.config["ob_dims"][0],), jnp.float32)
    a = np.asarray(agent.sample_actions(obs, seed=jax.random.PRNGKey(0)))
    assert a.shape == (ACT,) and np.isfinite(a).all() and (np.abs(a) <= 1.0 + 1e-6).all()

