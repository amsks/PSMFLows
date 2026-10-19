"""policy_index=reward: Density FB's z arm in psmgoal (docs/design/2026-10-19-reward-index.md).

Checks: the asserts fire; the index rows are sqrt(D)-sphere vectors with the stated hindsight share and
the same function gives the same w at the same key; the measure loss reads batch['index_w'] as online and
target rows; the bootstrap is the flowbc actor at (s', w_i); one update is finite, moves delta and the
basis, leaves w(z) and h alone, writes the index state; the eval path under coef_source=ridge produces
finite actions under acting=gpi (both ridge_score values) and acting=distill; the defaults are untouched.

Run in its own process: JAX_PLATFORMS=cpu .venv/bin/python -m pytest
tests/test_psmgoal_reward_index.py -q -p no:cacheprovider
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.psmgoal import FLOWBC_BOOT_KEY, PSMGoalAgent, get_config

OB, DA, Z, CODE, N = 6, 3, 8, 4, 8

RI = {"policy_index": "reward", "measure_loss": "softmax", "bootstrap_source": "actor", "train_actor": True,
      "actor_input": "w", "actor_kind": "flowbc", "fb_bc_coeff": 3.0, "coef_source": "ridge",
      "goal_random_frac": 0.0, "goal_cur_frac": 0.0, "acting": "gpi"}


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
    c.fb_k_anchor = 4
    c.fb_k_u = 2
    c.fb_hit_spec = "0:3:0.5"
    for k, v in over.items():
        setattr(c, k, v)
    return c


def _agent(seed=0, **over):
    return PSMGoalAgent.create(seed, np.zeros((2, OB), np.float32), np.zeros((2, DA), np.float32),
                               _cfg(**{**RI, **over}))


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


# ------------------------------------------------------------------ asserts
@pytest.mark.parametrize("bad", [
    {"measure_loss": "squared"}, {"train_actor": False}, {"actor_kind": "tanh"}, {"actor_input": "goal"},
    {"bootstrap_source": "data"}, {"coef_source": "hgoal_each"}, {"ridge_score": "x"}, {"norm_future_k": 2},
    {"acting": "sfbc"}, {"actor_value": "measure_reward_softmax"},
])
def test_reward_index_asserts(bad):
    with pytest.raises(AssertionError):
        _agent(**bad)


def test_defaults_untouched():
    c = get_config()
    assert c.policy_index == "code" and c.ridge_score == "goal" and c.coef_source == "lp"


# ------------------------------------------------------------------ index rows
def test_index_rows_shapes_and_sphere():
    ag = _agent(fb_rand_frac=0.0)
    rows = ag.reward_index_rows(_batch())
    assert rows["index_w"].shape == (N, Z) and rows["index_w_actor"].shape == (N, Z)
    assert rows["index_f"].shape == (N, Z) and rows["index_anchor"].shape == (4, OB)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(rows["index_w"]), axis=-1), np.sqrt(Z), rtol=1e-4)
    assert float(jnp.mean(rows["index_hindsight"])) == 1.0
    rows2 = ag.reward_index_rows(_batch())
    assert _leaves_equal(rows["index_w"], rows2["index_w"]), "same batch, same key -> same index rows"


def test_index_rows_random_share():
    ag = _agent(fb_rand_frac=1.0)
    rows = ag.reward_index_rows(_batch())
    assert float(jnp.mean(rows["index_hindsight"])) == 0.0
    np.testing.assert_allclose(np.linalg.norm(np.asarray(rows["index_w"]), axis=-1), np.sqrt(Z), rtol=1e-4)


def test_hindsight_index_equals_eval_solve_on_the_same_reward():
    """The training solve and the test-time solve are the same function at the same stored moments:
    the row's w from its kernel reward equals fb_task_fit on that reward."""
    from agents.psmgoal import kernel_reward, parse_hit_spec
    ag = _agent(fb_rand_frac=0.0)
    b = _batch()
    rows = ag.reward_index_rows(b)
    pool = jnp.asarray(b["next_observations"])
    r = kernel_reward(jnp.asarray(b["goals"]), pool, parse_hit_spec(ag.config["fb_hit_spec"]))
    w_eval = ag.fb_task_fit(rows["index_f"], r)
    np.testing.assert_allclose(np.asarray(rows["index_w"]), np.asarray(w_eval), atol=1e-5)


# ------------------------------------------------------------------ loss and bootstrap
def test_measure_loss_reads_index_rows_as_online_and_target():
    ag = _agent()
    b = _batch()
    b = {**b, **ag.reward_index_rows(b)}
    z = jnp.zeros((N, CODE))
    u_next = ag._select_u_next(b, z)
    loss, info = ag.measure_loss(ag.basis.params, ag.w.params, b, z, u_next)
    assert np.isfinite(float(loss))
    np.testing.assert_allclose(float(info["w_norm"]), np.sqrt(Z), rtol=1e-4)
    for k in ("index_proxy_corr", "index_w_norm", "index_hindsight_frac", "index_cov_trace", "measure_p_own_next"):
        assert k in info and np.isfinite(float(info[k])), k
    # w(z) gets no gradient: the index is not a net
    g_w = jax.grad(lambda p: ag.measure_loss(ag.basis.params, p, b, z, u_next)[0])(ag.w.params)
    assert all(np.allclose(x, 0.0) for x in jax.tree_util.tree_leaves(g_w))


def test_bootstrap_is_the_flowbc_actor_at_the_index_row():
    ag = _agent()
    b = _batch()
    b = {**b, **ag.reward_index_rows(b)}
    u_next = ag._select_u_next(b, jnp.zeros((N, CODE)))
    eps = jax.random.normal(jax.random.fold_in(ag.rng, FLOWBC_BOOT_KEY), (N, DA))
    expect = ag.flowbc_latent(jnp.asarray(b["next_observations"]), jnp.asarray(b["index_w"]), eps)[0]
    np.testing.assert_allclose(np.asarray(u_next), np.asarray(expect), atol=1e-6)
    assert u_next.shape == (N, DA) and float(jnp.max(jnp.abs(u_next))) <= float(ag.config["u_clip"])


# ------------------------------------------------------------------ update
def test_update_moves_delta_and_basis_only_and_writes_index_state():
    ag = _agent()
    d0, b0, w0, h0, a0 = ag.delta.params, ag.basis.params, ag.w.params, ag.w_star.params, ag.actor.params
    new, info = ag.update(_batch())
    for k in ("psm_loss", "actor_loss", "actor_delta_norm", "index_proxy_corr", "actor_p_own", "index_hindsight_frac_actor"):
        assert k in info and np.isfinite(float(info[k])), k
    assert not _leaves_equal(new.delta.params, d0)
    assert not _leaves_equal(new.basis.params, b0)
    assert _leaves_equal(new.w.params, w0) and _leaves_equal(new.w_star.params, h0) and _leaves_equal(new.actor.params, a0)
    assert float(new.fb_filled) == 1.0
    # the index tower EMAs toward the PRE-step basis: after one update it still equals the init copy,
    # after two it has moved
    assert _leaves_equal(new.fb_zenc, ag.fb_zenc)
    assert not np.allclose(np.asarray(new.fb_mu), 0.0)
    new2, info2 = new.update(_batch(seed=1))
    assert np.isfinite(float(info2["psm_loss"]))
    assert not _leaves_equal(new2.fb_zenc, ag.fb_zenc)
    assert _leaves_equal(new2.fb_anchor, new.fb_anchor), "anchors are fixed after the first batch"


# ------------------------------------------------------------------ eval
def _relabel(n=16, seed=3):
    rng = np.random.default_rng(seed)
    b = _batch(n=n, seed=seed)
    r = -np.ones((n,), np.float32)
    r[rng.permutation(n)[:6]] = 0.0            # six rewarding rows after the +1 shift
    return b, r + 1.0


@pytest.mark.parametrize("ridge_score", ["goal", "reward"])
def test_eval_gpi_at_the_ridge_w(ridge_score):
    ag = _agent(ridge_score=ridge_score)
    ag, _ = ag.update(_batch())
    b, r = _relabel()
    ag = ag.infer_eval_goals(b, r)
    np.testing.assert_allclose(float(jnp.linalg.norm(ag.eval_w_star)), np.sqrt(Z), rtol=1e-4)
    if ridge_score == "reward":
        np.testing.assert_allclose(np.asarray(ag.eval_cand_w), np.broadcast_to(np.asarray(ag.eval_w_star)[None], (4, Z)))
        assert set(np.unique(np.asarray(ag.eval_ref_reward))) <= {0.0, 1.0}
    a = ag.sample_actions(jnp.asarray(b["observations"][0]), seed=jax.random.PRNGKey(0))
    assert a.shape == (DA,) and np.all(np.isfinite(np.asarray(a)))


def test_eval_distill_acts_with_the_flowbc_actor_at_w():
    ag = _agent(acting="distill", eval_redistill=False)
    ag, _ = ag.update(_batch())
    b, r = _relabel()
    ag = ag.infer_eval_goals(b, r)
    obs = jnp.asarray(b["observations"][0])
    a = ag.sample_actions(obs, seed=jax.random.PRNGKey(1))
    assert a.shape == (DA,) and np.all(np.isfinite(np.asarray(a)))
    eps = jax.random.normal(jax.random.PRNGKey(1), (1, DA))
    u = ag.flowbc_latent(obs[None], ag.eval_w_star[None], eps)[0][0]
    np.testing.assert_allclose(np.asarray(a), np.asarray(ag.decode(obs[None], u[None])[0]), atol=1e-6)
