"""psmgoal, best-of-N bootstrap with no actor (bootstrap_source=argmax, 2026-10-07).

u'_i is the best of bootstrap_n clipped prior draws at s'_i, scored by the target measure on the
row's goal (softmax: the share of g_i among [g_i + the batch's next states]). Checks: the new keys
default off in both configs; bootstrap_n=1 returns the clipped prior draw; the chosen latent is the
argmax of a hand-computed score (both losses); an update trains the measure and h and never the
actor; the spec's run configs build with their readouts; the `create` asserts fire.

Run in its own process: JAX_PLATFORMS=cpu .venv/bin/python -m pytest
tests/test_psmgoal_argmax_bootstrap.py -q -p no:cacheprovider
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import agents.psmgoal as pg
from agents.psmgoal import PSMGoalAgent, get_config

OB, DA, Z, CODE, N = 6, 3, 8, 4, 8

# The arm of the 2026-10-07 run: the 0.80 recipe (psmgoal_ja_fbc3_pt_cube) with the actor removed.
BON = {"policy_index": "goal", "bootstrap_source": "argmax", "train_actor": False,
       "goal_random_frac": 0.0, "goal_cur_frac": 0.0, "goal_sampling": "geometric"}
ARMS = {"sq": {**BON, "measure_loss": "squared"}, "sm": {**BON, "measure_loss": "softmax"}}


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


def _agent(seed=0, **over):
    return PSMGoalAgent.create(seed, np.zeros((2, OB), np.float32), np.zeros((2, DA), np.float32),
                               _cfg(**over))


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


def _leaves(tree):
    return jax.tree_util.tree_leaves(tree)


def _same(t0, t1):
    return all(jnp.array_equal(a, b) for a, b in zip(_leaves(t0), _leaves(t1)))


def _changed(t0, t1):
    return any(not jnp.array_equal(a, b) for a, b in zip(_leaves(t0), _leaves(t1)))


def _code(seed):
    return jax.random.bernoulli(jax.random.PRNGKey(seed), 0.5, (N, CODE)).astype(jnp.float32)


def _draws(ag, n):
    eps = jax.random.normal(jax.random.fold_in(ag.rng, pg.ARGMAX_BOOT_KEY), (n, N, DA))
    return np.clip(np.asarray(eps), -3.0, 3.0)


def test_new_keys_default_off_in_get_config_and_yaml():
    c = get_config()
    assert c.bootstrap_source == "actor"
    assert c.bootstrap_n == 8
    y = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "agent", "psmgoal.yaml")))
    assert y["bootstrap_source"] == "actor"
    assert y["bootstrap_n"] == 8
    assert "argmax" in pg.BOOTSTRAP_SOURCES


@pytest.mark.parametrize("arm", ["sq", "sm"])
def test_bootstrap_n1_is_the_clipped_prior_draw(arm):
    ag = _agent(**ARMS[arm], bootstrap_n=1)
    b = _batch()
    u_sel = np.asarray(ag._select_u_next(b, _code(0)))
    np.testing.assert_array_equal(u_sel, _draws(ag, 1)[0])


@pytest.mark.parametrize("arm", ["sq", "sm"])
def test_bootstrap_is_the_argmax_of_the_hand_score(arm):
    ag = _agent(**ARMS[arm], bootstrap_n=5, measure_temp=0.7)
    ag, _ = ag.update(_batch(seed=1))                         # move the target off its init
    b = _batch()
    eps = _draws(ag, 5)
    nobs, g = jnp.asarray(b["next_observations"]), jnp.asarray(b["goals"])
    w_t = ag.w_star(g, params=ag.target_w_star)
    hand = np.zeros((5, N))
    for k in range(5):
        u = jnp.asarray(eps[k])
        m_g = np.asarray(ag.M(nobs, u, g, w_t, params=ag.target_basis))
        if arm == "sq":
            hand[k] = m_g
        else:
            M = np.asarray(ag.mesh_M(nobs, u, nobs, w_t, params=ag.target_basis))
            logits = np.concatenate([m_g[:, None], M], axis=1) / 0.7
            logits -= logits.max(1, keepdims=True)
            hand[k] = np.exp(logits[:, 0]) / np.exp(logits).sum(1)
    u_sel, scores = ag.argmax_bootstrap(b)
    np.testing.assert_allclose(np.asarray(scores), hand, rtol=1e-4, atol=1e-6)
    want = eps[hand.argmax(0), np.arange(N)]
    np.testing.assert_array_equal(np.asarray(u_sel), want)
    np.testing.assert_array_equal(np.asarray(ag._select_u_next(b, _code(0))), want)


@pytest.mark.parametrize("arm", ["sq", "sm"])
def test_update_trains_measure_and_h_but_never_the_actor(arm):
    ag = _agent(**ARMS[arm], bootstrap_n=4)
    for step in range(2):
        new, info = ag.update(_batch(seed=step))
        assert _same(ag.actor.params, new.actor.params), "no actor step under bootstrap_source=argmax"
        assert _same(ag.delta.params, new.delta.params)
        assert not any(k.startswith("actor") for k in info), sorted(info)
        assert _changed(ag.basis.params, new.basis.params)
        assert _changed(ag.w_star.params, new.w_star.params)
        assert np.isfinite(float(info["psm_loss"]))
        ag = new


def test_run_configs_build_with_their_readouts():
    for arm in ARMS.values():
        for n in (1, 8, 64):
            _agent(**arm, bootstrap_n=n)
            _agent(**arm, bootstrap_n=n, coef_source="hgoal_each", acting="gpi")


@pytest.mark.parametrize("over", [
    {"bootstrap_source": "argmax"},                                     # no goal index
    {**BON, "train_actor": True, "actor_input": "goal"},                # argmax replaces the actor
    {**BON, "bootstrap_n": 0},
])
def test_create_asserts_fire(over):
    with pytest.raises(AssertionError):
        _agent(**over)
