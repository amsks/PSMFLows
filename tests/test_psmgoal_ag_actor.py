"""ag_actor: the advantage-guided actor in psmgoal (docs/design/2026-10-20-advantage-guided-actor.md).

Checks: the asserts fire; the defaults are untouched; delta's cond slot is [g, c]; the per-row scores are
standardised; the TD bootstrap equals clip(eps + ag_w_boot (mu(+1) - mu(-1))) read from the slow copy; one
update moves delta and moves delta_target toward it by ag_delta_tau and leaves the goal head alone; acting
under distill uses ag_w_eval and two weights give two actions; the supervised loss falls over a few steps on a
fixed batch.

Run in its own process: JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_psmgoal_ag_actor.py -q -p no:cacheprovider
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.psmgoal import AG_DRAW_KEY, FLOWBC_BOOT_KEY, PSMGoalAgent, get_config

OB, DA, Z, CODE, N = 6, 3, 8, 4, 8

AG = {"policy_index": "goal", "measure_loss": "softmax", "bootstrap_source": "actor", "train_actor": True,
      "actor_input": "goal", "actor_kind": "flowbc", "actor_value_kind": "point", "ag_actor": True, "ag_k": 6,
      "goal_random_frac": 0.0, "goal_cur_frac": 0.0, "acting": "distill", "eval_redistill": False,
      "eval_goal_source": "relabel"}


def _cfg(**over):
    c = get_config()
    c.z_dim = Z
    c.max_log_seed = CODE
    c.batch_size = N
    c.k_goals = 4
    c.gpi_num_u = 5
    c.num_inference_steps = 2
    c.infer_batch = N
    c.allow_untrained_flow = True
    c.measure.hidden_dim = 16
    c.w.hidden_dim = 16
    c.l.hidden_dim = 16
    c.actor.hidden_dim = 16
    c.flow.hidden_dims = (16, 16)
    c.fb_hit_spec = "0:3:0.5"
    for k, v in over.items():
        setattr(c, k, v)
    return c


def _agent(seed=0, **over):
    return PSMGoalAgent.create(seed, np.zeros((2, OB), np.float32), np.zeros((2, DA), np.float32),
                               _cfg(**{**AG, **over}))


def _batch(n=N, seed=0):
    rng = np.random.default_rng(seed)
    nxt = rng.standard_normal((n, OB)).astype(np.float32)
    return {
        "observations": rng.standard_normal((n, OB)).astype(np.float32),
        "next_observations": nxt,
        "noise_preimage": rng.standard_normal((n, DA)).astype(np.float32),
        "goals": nxt[rng.permutation(n)],
        "fb_goals": nxt[rng.permutation(n)],
        "index": np.arange(n).astype(np.int32),
        "rewards": -np.ones((n,), np.float32),
    }


def _leaves_equal(a, b):
    return all(np.allclose(x, y) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


@pytest.mark.parametrize("bad", [
    {"actor_kind": "tanh"}, {"actor_input": "w"}, {"bootstrap_source": "data"}, {"actor_value_kind": "shaped"},
    {"train_actor": False}, {"bootstrap_delta_scale": 0.5}, {"ag_k": 1}, {"ag_delta_tau": 0.0},
])
def test_ag_asserts(bad):
    with pytest.raises(AssertionError):
        _agent(**bad)


def test_defaults_untouched():
    c = get_config()
    assert c.ag_actor is False and c.ag_k == 64 and c.ag_w_boot == 0.5 and c.ag_w_eval == 1.0 and c.ag_delta_tau == 0.005
    ag = PSMGoalAgent.create(0, np.zeros((2, OB), np.float32), np.zeros((2, DA), np.float32), _cfg())
    assert ag.delta_target is None


def test_mu_reads_goal_and_score_and_shift_is_the_difference():
    ag = _agent()
    b = _batch()
    obs, g = jnp.asarray(b["observations"]), jnp.asarray(b["fb_goals"])
    ones = jnp.ones((N,))
    mu_p, mu_m = ag.ag_mu(obs, g, ones), ag.ag_mu(obs, g, -ones)
    assert mu_p.shape == (N, DA)
    np.testing.assert_allclose(np.asarray(ag.ag_shift(obs, g, 0.5)), 0.5 * np.asarray(mu_p - mu_m), atol=1e-6)
    # at init the output layer is zero, so the shift is zero and the bootstrap is the BC draw
    assert np.allclose(np.asarray(mu_p), 0.0)


def test_bootstrap_is_the_guided_draw_from_the_slow_copy():
    ag = _agent()
    ag, _ = ag.update(_batch())
    ag, _ = ag.update(_batch(seed=1))
    b = _batch(seed=2)
    z = jnp.zeros((N, CODE), jnp.int32)
    u = ag._select_u_next(b, z)
    eps = jax.random.normal(jax.random.fold_in(ag.rng, FLOWBC_BOOT_KEY), (N, DA))
    want = ag.ag_latent(jnp.asarray(b["next_observations"]), jnp.asarray(b["goals"]), eps,
                        float(ag.config["ag_w_boot"]), params=ag.delta_target)[0]
    np.testing.assert_allclose(np.asarray(u), np.asarray(want), atol=1e-6)
    other = ag.ag_latent(jnp.asarray(b["next_observations"]), jnp.asarray(b["goals"]), eps,
                         float(ag.config["ag_w_boot"]), params=ag.delta.params)[0]
    assert not np.allclose(np.asarray(u), np.asarray(other)), "the bootstrap reads the slow copy, not delta"


def test_update_moves_delta_and_polyaks_the_copy_and_leaves_the_goal_head_alone():
    ag = _agent()
    d0, t0, h0, b0 = ag.delta.params, ag.delta_target, ag.w_star.params, ag.basis.params
    new, info = ag.update(_batch())
    for k in ("psm_loss", "actor_loss", "actor_delta_norm", "actor_q", "ag_q_gain", "ag_q_gain_bok", "ag_c_absmean", "ag_shift_cos_best"):
        assert k in info and np.isfinite(float(info[k])), k
    assert not _leaves_equal(new.delta.params, d0)
    assert not _leaves_equal(new.basis.params, b0)
    tau = float(ag.config["ag_delta_tau"])
    want = jax.tree_util.tree_map(lambda p, t: p * tau + t * (1.0 - tau), new.delta.params, t0)
    assert _leaves_equal(new.delta_target, want)
    # the goal head moves only through the measure loss, as without ag_actor
    ref = PSMGoalAgent.create(0, np.zeros((2, OB), np.float32), np.zeros((2, DA), np.float32),
                              _cfg(**{**AG, "ag_actor": False}))
    assert _leaves_equal(ref.w_star.params, h0)
    # standardised scores: |c| averages near 1 for a non-degenerate score (K = 6 draws)
    assert 0.3 < float(info["ag_c_absmean"]) < 1.5


def test_scores_are_standardised_per_row():
    ag = _agent()
    b = _batch()
    obs, pool = jnp.asarray(b["observations"]), jnp.asarray(b["next_observations"])
    goals = jnp.asarray(b["fb_goals"])
    K = int(ag.config["ag_k"])
    eps = jax.random.normal(jax.random.fold_in(ag.rng, AG_DRAW_KEY), (K, N, DA))
    u_rows = jnp.clip(eps, -ag.config["u_clip"], ag.config["u_clip"])
    w_rows = ag.w_star(goals)
    q = jnp.stack([ag.actor_value(obs, u_rows[k], pool, goals, w_rows, ag.basis.params) for k in range(K)])
    c = (q - q.mean(0)) / jnp.sqrt(q.var(0) + 1e-6)
    np.testing.assert_allclose(np.asarray(c.mean(0)), 0.0, atol=1e-4)
    assert np.all(np.asarray(c.std(0)) < 1.0 + 1e-3)


def test_supervised_loss_falls_on_fixed_draws_and_scores():
    """Actor steps alone (critic and rng held, so the K draws and their scores are the same every step)."""
    ag = _agent()
    b = _batch()
    losses = []
    for _ in range(8):
        upd, info = ag.ag_actor_step(b)
        ag = ag.replace(**upd)
        losses.append(float(info["actor_loss"]))
    assert losses[-1] < losses[0], losses


def _relabel(n=16, seed=3):
    rng = np.random.default_rng(seed)
    b = _batch(n=n, seed=seed)
    r = -np.ones((n,), np.float32)
    r[rng.permutation(n)[:6]] = 0.0
    return b, r + 1.0


def test_acting_uses_the_guidance_weight():
    ag = _agent()
    for s in range(3):
        ag, _ = ag.update(_batch(seed=s))
    b, r = _relabel()
    ag = ag.infer_eval_goals(b, r)
    obs = jnp.asarray(b["observations"][0])
    a1 = ag.sample_actions(obs, seed=jax.random.PRNGKey(1))
    assert a1.shape == (DA,) and np.all(np.isfinite(np.asarray(a1)))
    eps = jax.random.normal(jax.random.PRNGKey(1), (1, DA))
    u = ag.ag_latent(obs[None], ag.eval_actor_goal[None], eps, float(ag.config["ag_w_eval"]))[0][0]
    np.testing.assert_allclose(np.asarray(a1), np.asarray(ag.decode(obs[None], u[None])[0]), atol=1e-6)
    ag4 = ag.replace(config=ag.config.copy({"ag_w_eval": 4.0}))
    a4 = ag4.sample_actions(obs, seed=jax.random.PRNGKey(1))
    assert not np.allclose(np.asarray(a1), np.asarray(a4))
    ag0 = ag.replace(config=ag.config.copy({"ag_w_eval": 0.0}))
    a0 = ag0.sample_actions(obs, seed=jax.random.PRNGKey(1))
    np.testing.assert_allclose(np.asarray(a0), np.asarray(ag.decode(obs[None], jnp.clip(eps, -ag.config["u_clip"], ag.config["u_clip"]))[0]), atol=1e-6)


def test_ag_pool_subset_runs_and_scores_the_same_draws():
    """ag_pool=3: the update runs and the subset score is a coarser estimate of the full-pool share:
    finite, in (0, 1), and positively correlated with the full-pool score over draws and rows."""
    ag = _agent(ag_pool=3)
    new, info = ag.update(_batch())
    assert np.isfinite(float(info["actor_loss"])) and np.isfinite(float(info["ag_c_absmean"]))
    b = _batch()
    obs, pool, goals = jnp.asarray(b["observations"]), jnp.asarray(b["next_observations"]), jnp.asarray(b["fb_goals"])
    eps = jax.random.normal(jax.random.PRNGKey(0), (4, N, DA))
    w_rows = new.w_star(goals)
    full = np.asarray(jnp.stack([new.actor_value(obs, eps[k], pool, goals, w_rows, new.basis.params) for k in range(4)]))
    sub = np.asarray(jnp.stack([new.actor_value(obs, eps[k], pool[:3], goals, w_rows, new.basis.params) for k in range(4)]))
    assert np.all((sub > 0) & (sub < 1)) and np.all(np.isfinite(sub))
    assert np.corrcoef(full.ravel(), sub.ravel())[0, 1] > 0.5
