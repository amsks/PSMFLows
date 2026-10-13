"""psmgoal, average-of-N BC bootstrap with no actor (bootstrap_source=bc_mean, 2026-10-13).

The target at s'_i is the target measure averaged over bootstrap_n clipped prior draws (softmax:
the mean of the draws' softmax rows). Checks: the key defaults off; the bootstrap is (n, N, d) and
the target (N, N); the target is the hand-computed mean for both losses; bootstrap_n=1 gives the
same loss and the same update as bootstrap_source=argmax with bootstrap_n=1 (one BC draw, bon1); an
update never moves the actor; the `create` asserts fire.

Run in its own process: JAX_PLATFORMS=cpu .venv/bin/python -m pytest
tests/test_psmgoal_bc_mean_bootstrap.py -q -p no:cacheprovider
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import agents.psmgoal as pg
from agents.psmgoal import PSMGoalAgent, get_config

OB, DA, Z, CODE, N = 6, 3, 8, 4, 8

ARM = {"policy_index": "goal", "train_actor": False, "goal_random_frac": 0.0, "goal_cur_frac": 0.0,
       "goal_sampling": "geometric"}
LOSSES = {"sq": "squared", "sm": "softmax"}


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
    c.w_star.hidden_dim = 16
    c.flow.hidden_dims = (16, 16)
    c.fb_hit_spec = "0:3:0.5"
    for k, v in over.items():
        setattr(c, k, v)
    return c


def _agent(loss, source="bc_mean", n=4, seed=0, **over):
    cfg = _cfg(**ARM, measure_loss=LOSSES[loss], bootstrap_source=source, bootstrap_n=n, **over)
    return PSMGoalAgent.create(seed, np.zeros((2, OB), np.float32), np.zeros((2, DA), np.float32), cfg)


def _batch(n=N, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "observations": rng.standard_normal((n, OB)).astype(np.float32),
        "next_observations": rng.standard_normal((n, OB)).astype(np.float32),
        "noise_preimage": rng.standard_normal((n, DA)).astype(np.float32),
        "goals": rng.standard_normal((n, OB)).astype(np.float32),
        "index": np.arange(n).astype(np.int32),
        "rewards": -np.ones((n,), np.float32),
    }


def _code(seed=0):
    return jax.random.bernoulli(jax.random.PRNGKey(seed), 0.5, (N, CODE)).astype(jnp.float32)


def _leaves(t):
    return jax.tree_util.tree_leaves(t)


def test_key_defaults_off():
    assert get_config().bootstrap_source == "actor"
    assert "bc_mean" in pg.BOOTSTRAP_SOURCES


@pytest.mark.parametrize("loss", ["sq", "sm"])
def test_bootstrap_and_target_shapes(loss):
    ag = _agent(loss, n=5)
    b = _batch()
    eps = ag._select_u_next(b, _code())
    assert eps.shape == (5, N, DA)
    assert float(jnp.max(jnp.abs(eps))) <= float(ag.config["u_clip"])
    w_t = ag.w_star(jnp.asarray(b["goals"]), params=ag.target_w_star)
    t = ag.bc_mean_target(jnp.asarray(b["next_observations"]), eps, jnp.asarray(b["next_observations"]), w_t)
    assert t.shape == (N, N)
    assert np.all(np.isfinite(np.asarray(t)))


@pytest.mark.parametrize("loss", ["sq", "sm"])
def test_target_is_the_hand_mean(loss):
    ag = _agent(loss, n=4, measure_temp=0.7)
    ag, _ = ag.update(_batch(seed=1))                       # move the target off its init
    b = _batch()
    nobs = jnp.asarray(b["next_observations"])
    eps = ag._select_u_next(b, _code())
    w_t = ag.w_star(jnp.asarray(b["goals"]), params=ag.target_w_star)
    Ms = np.stack([np.asarray(ag.mesh_M(nobs, eps[k], nobs, w_t, params=ag.target_basis)) for k in range(4)])
    got = np.asarray(ag.bc_mean_target(nobs, eps, nobs, w_t))
    if loss == "sq":
        np.testing.assert_allclose(got, Ms.mean(0), rtol=1e-5, atol=1e-6)
    else:
        def softmax(x):
            x = x - x.max(-1, keepdims=True)
            return np.exp(x) / np.exp(x).sum(-1, keepdims=True)
        want = np.mean([softmax(Ms[k] / 0.7) for k in range(4)], axis=0)
        np.testing.assert_allclose(softmax(got / 0.7), want, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("loss", ["sq", "sm"])
def test_n1_is_bon1(loss):
    """bootstrap_n=1: the same BC draw as argmax/bon1, so the same loss and the same update."""
    a_mean, a_max = _agent(loss, "bc_mean", n=1), _agent(loss, "argmax", n=1)
    for x, y in zip(_leaves(a_mean.basis.params), _leaves(a_max.basis.params)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    b = _batch()
    np.testing.assert_array_equal(np.asarray(a_mean._select_u_next(b, _code())[0]),
                                  np.asarray(a_max._select_u_next(b, _code())))
    l_mean = float(a_mean.total_loss(b)[0])
    l_max = float(a_max.total_loss(b)[0])
    np.testing.assert_allclose(l_mean, l_max, rtol=1e-5)
    n_mean, _ = a_mean.update(b)
    n_max, _ = a_max.update(b)
    for x, y in zip(_leaves(n_mean.basis.params), _leaves(n_max.basis.params)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("loss", ["sq", "sm"])
def test_update_never_moves_the_actor(loss):
    ag = _agent(loss, n=4)
    for step in range(2):
        new, info = ag.update(_batch(seed=step))
        assert all(jnp.array_equal(x, y) for x, y in zip(_leaves(ag.actor.params), _leaves(new.actor.params)))
        assert not any(k.startswith("actor") for k in info), sorted(info)
        assert any(not jnp.array_equal(x, y) for x, y in zip(_leaves(ag.basis.params), _leaves(new.basis.params)))
        assert np.isfinite(float(info["psm_loss"]))
        ag = new


@pytest.mark.parametrize("over", [
    {"policy_index": "code"},                                # needs h(g)
    {"train_actor": True, "actor_input": "goal"},            # replaces the actor
    {"bootstrap_n": 0},
])
def test_create_asserts_fire(over):
    cfg = _cfg(**{**ARM, "bootstrap_source": "bc_mean", "bootstrap_n": 4, **over})
    with pytest.raises(AssertionError):
        PSMGoalAgent.create(0, np.zeros((2, OB), np.float32), np.zeros((2, DA), np.float32), cfg)
