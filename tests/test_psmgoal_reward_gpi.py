"""psmgoal coef_source=reward_gpi (2026-10-15, eval only): the task is a reward on states
r(s+). Each candidate coefficient h(g_c) is read as the expected reward over the reference
states and the acting score is the max over candidates.

Checks: the key defaults in both configs; `_reward_gpi_score` has shape (N,) and is finite;
with one candidate it equals the hand-computed softmax-share-weighted reward from
`mesh_phi_b`; a constant added to the reference reward adds that constant to every score;
the max over candidates is >= each single candidate's score; chunked equals unchunked;
`infer_eval_goals` with continuous rewards sets eval_cand_w and eval_ref_reward and
reward_gpi_cands=top picks the top-k rows; `select_latent` returns a latent in the clip box;
the squared-loss readout matches its hand computation; the `create` asserts fire.

Run in its own process: JAX_PLATFORMS=cpu .venv/bin/python -m pytest
tests/test_psmgoal_reward_gpi.py -q -p no:cacheprovider
"""
import os
import pickle

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import agents.psmgoal as pg
from agents.psmgoal import PSMGoalAgent, get_config, reward_gpi_candidate_rows
from utils.flax_utils import restore_agent

OB, DA, Z, CODE, N = 6, 3, 8, 4, 8
K_GOALS = 4

DB = {"policy_index": "goal", "bootstrap_source": "data", "train_actor": False,
      "goal_random_frac": 0.0, "goal_cur_frac": 0.0}
ARMS = {"sq": {**DB, "measure_loss": "squared"}, "sm": {**DB, "measure_loss": "softmax"}}
READOUT = {"coef_source": "reward_gpi", "acting": "gpi"}


def _cfg(**over):
    c = get_config()
    c.z_dim = Z
    c.max_log_seed = CODE
    c.batch_size = N
    c.k_goals = K_GOALS
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
        "next_noise_preimage": rng.standard_normal((n, DA)).astype(np.float32),
        "goals": rng.standard_normal((n, OB)).astype(np.float32),
        "index": np.arange(n).astype(np.int32),
        "rewards": -np.ones((n,), np.float32),
    }


def _relabel(n=32, seed=5):
    """A relabel batch with a continuous state reward: the next state's first coordinate."""
    b = _batch(n=n, seed=seed)
    return b, b["next_observations"][:, 0].astype(np.float32)


def _trained(arm, **over):
    ag = _agent(**{**ARMS[arm], **READOUT, **over})
    ag, _ = ag.update(_batch(seed=1))
    return ag


def _np_softmax(x, axis=-1):
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


def _rows(seed=4):
    rng = np.random.default_rng(seed)
    obs = jnp.asarray(rng.standard_normal((5, OB)), jnp.float32)
    u = jnp.asarray(rng.standard_normal((5, DA)), jnp.float32)
    refs = jnp.asarray(rng.standard_normal((6, OB)), jnp.float32)
    r = jnp.asarray(rng.standard_normal((6,)), jnp.float32)
    cands = jnp.asarray(rng.standard_normal((3, OB)), jnp.float32)
    return obs, u, refs, r, cands


# ------------------------------------------------------------------ defaults

def test_key_defaults_in_get_config_and_yaml():
    c = get_config()
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "agent", "psmgoal.yaml")
    with open(path) as f:
        y = yaml.safe_load(f)
    assert c["reward_gpi_cands"] == "both" and y["reward_gpi_cands"] == "both"
    assert "reward_gpi" in pg.GOAL_COEF_SOURCES and "reward_gpi" not in pg.COEF_SOURCES


def test_fresh_agent_has_the_eval_fields():
    ag = _agent(**{**ARMS["sm"], **READOUT})
    assert ag.eval_cand_w.shape == (K_GOALS, Z)
    assert ag.eval_ref_reward.shape == (N - K_GOALS,)
    assert ag.eval_ref.shape == (N - K_GOALS, OB)


# ------------------------------------------------------------------ the score

def test_score_shape_and_finite():
    ag = _trained("sm")
    obs, u, refs, r, cands = _rows()
    q = ag._reward_gpi_score(obs, u, ag.w_star(cands), refs, r)
    assert q.shape == (5,) and np.all(np.isfinite(np.asarray(q)))


def test_one_candidate_softmax_matches_hand():
    """value = sum_r softmax_r((phi_r h + b_r) / t) r_r."""
    ag = _trained("sm", measure_temp=0.7)
    obs, u, refs, r, cands = _rows()
    h = ag.w_star(cands[:1])                                    # (1, Z)
    phi, b = ag.mesh_phi_b(obs, u, refs)                        # (5, 6, Z), (5, 6)
    m = np.asarray(phi) @ np.asarray(h)[0] + np.asarray(b)      # (5, 6)
    hand = (_np_softmax(m / 0.7) * np.asarray(r)[None]).sum(1)
    got = ag._reward_gpi_score(obs, u, h, refs, r)
    np.testing.assert_allclose(np.asarray(got), hand, rtol=1e-4, atol=1e-6)


def test_one_candidate_squared_matches_hand():
    """value = mean_r (phi_r h + b_r) r_r."""
    ag = _trained("sq")
    obs, u, refs, r, cands = _rows()
    h = ag.w_star(cands[:1])
    phi, b = ag.mesh_phi_b(obs, u, refs)
    m = np.asarray(phi) @ np.asarray(h)[0] + np.asarray(b)
    hand = (m * np.asarray(r)[None]).mean(1)
    got = ag._reward_gpi_score(obs, u, h, refs, r)
    np.testing.assert_allclose(np.asarray(got), hand, rtol=1e-4, atol=1e-6)


def test_softmax_score_shifts_with_a_constant_reward_shift():
    """The shares sum to one, so r + c gives score + c for every row and candidate."""
    ag = _trained("sm")
    obs, u, refs, r, cands = _rows()
    h = ag.w_star(cands)
    base = np.asarray(ag._reward_gpi_values(obs, u, h, refs, r))
    shifted = np.asarray(ag._reward_gpi_values(obs, u, h, refs, r + 2.5))
    np.testing.assert_allclose(shifted, base + 2.5, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(ag._reward_gpi_score(obs, u, h, refs, r + 2.5)),
                               np.asarray(ag._reward_gpi_score(obs, u, h, refs, r)) + 2.5, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("arm", ["sq", "sm"])
def test_max_over_candidates_dominates_each_single_candidate(arm):
    ag = _trained(arm)
    obs, u, refs, r, cands = _rows()
    h = ag.w_star(cands)
    q = np.asarray(ag._reward_gpi_score(obs, u, h, refs, r))
    singles = np.stack([np.asarray(ag._reward_gpi_score(obs, u, h[c:c + 1], refs, r)) for c in range(3)], 1)
    assert np.all(q >= singles.max(1) - 1e-6)
    np.testing.assert_allclose(q, singles.max(1), rtol=1e-5, atol=1e-6)
    idx = np.asarray(ag._reward_gpi_argmax(obs, u, h, refs, r))
    np.testing.assert_array_equal(idx, singles.argmax(1))


@pytest.mark.parametrize("arm", ["sq", "sm"])
def test_chunked_equals_unchunked(arm):
    ag = _trained(arm)
    rng = np.random.default_rng(6)
    obs = jnp.asarray(rng.standard_normal((5, OB)), jnp.float32)
    u = jnp.asarray(rng.standard_normal((5, DA)), jnp.float32)
    refs = jnp.asarray(rng.standard_normal((6, OB)), jnp.float32)
    r = jnp.asarray(rng.standard_normal((6,)), jnp.float32)
    h = ag.w_star(jnp.asarray(rng.standard_normal((11, OB)), jnp.float32))
    full = ag._reward_gpi_values(obs, u, h, refs, r)
    chunked = ag._reward_gpi_values(obs, u, h, refs, r, chunk=4)
    np.testing.assert_allclose(np.asarray(chunked), np.asarray(full), rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------------ infer_eval_goals

def test_candidate_rows_by_mode():
    rewards = np.array([0.1, 0.9, 0.5, 0.7, 0.3, 0.8], np.float32)
    np.testing.assert_array_equal(reward_gpi_candidate_rows(rewards, 4, "top"), [1, 5, 3, 2])
    np.random.seed(0)
    rand = reward_gpi_candidate_rows(rewards, 4, "random")
    assert rand.shape == (4,) and len(set(rand.tolist())) == 4
    both = reward_gpi_candidate_rows(rewards, 4, "both")
    assert both.shape == (4,) and len(set(both.tolist())) == 4
    assert set(both[:2].tolist()) == {1, 5}
    assert not (set(both[2:].tolist()) & {1, 5})


@pytest.mark.parametrize("arm", ["sq", "sm"])
def test_infer_sets_candidates_and_reference_rewards(arm):
    ag = _trained(arm, reward_gpi_cands="top")
    b, rew = _relabel()
    np.random.seed(0)
    ev = ag.infer_eval_goals(b, rew)
    assert ev.eval_cand_w.shape == (K_GOALS, Z)
    assert ev.eval_ref_reward.shape == (N - K_GOALS,)
    assert ev.eval_ref.shape == (N - K_GOALS, OB)
    top = np.argsort(-rew, kind="stable")[:K_GOALS]
    np.testing.assert_allclose(np.asarray(ev.eval_goals), b["next_observations"][top], atol=1e-7)
    np.testing.assert_allclose(np.asarray(ev.eval_cand_w), np.asarray(ev.w_star(ev.eval_goals)), atol=1e-6)
    np.testing.assert_allclose(np.asarray(ev.eval_goal_w), np.asarray(ev.eval_cand_w), atol=1e-7)
    # the reference rewards are the rewards of the reference states
    refs = np.asarray(ev.eval_ref)
    for j in range(refs.shape[0]):
        i = np.flatnonzero(np.all(np.isclose(b["next_observations"], refs[j]), axis=1))
        assert i.size == 1 and np.isclose(float(ev.eval_ref_reward[j]), rew[i[0]])
    assert np.all(np.asarray(ev.eval_w_star) == 0.0)


def test_infer_runs_with_both_and_random_candidates():
    for mode in ("both", "random"):
        ag = _trained("sm", reward_gpi_cands=mode)
        b, rew = _relabel()
        ev = ag.infer_eval_goals(b, rew)
        assert ev.eval_goals.shape == (K_GOALS, OB) and np.all(np.isfinite(np.asarray(ev.eval_cand_w)))


def test_infer_takes_negative_and_fractional_rewards():
    ag = _trained("sm")
    b, rew = _relabel()
    ev = ag.infer_eval_goals(b, rew - 10.0)
    assert np.all(np.asarray(ev.eval_ref_reward) < 0.0)


# ------------------------------------------------------------------ acting

@pytest.mark.parametrize("arm", ["sq", "sm"])
def test_select_latent_is_the_argmax_prior_draw_in_the_clip_box(arm):
    ag = _trained(arm)
    b, rew = _relabel()
    ev = ag.infer_eval_goals(b, rew)
    obs0 = jnp.asarray(b["observations"][0])
    seed = jax.random.PRNGKey(4)
    K = int(ev.config["gpi_num_u"])
    u_cand = jnp.clip(jax.random.normal(seed, (K, DA)), -3.0, 3.0)
    obs = jnp.broadcast_to(obs0, (K, OB))
    q = ev._reward_gpi_score(obs, u_cand, ev.eval_cand_w, ev.eval_ref, ev.eval_ref_reward)
    u_star = ev.select_latent(obs0, seed)
    assert u_star.shape == (DA,) and float(jnp.max(jnp.abs(u_star))) <= 3.0
    np.testing.assert_array_equal(np.asarray(u_star), np.asarray(u_cand[int(np.argmax(q))]))
    a = ev.sample_actions(obs0, seed=jax.random.PRNGKey(1))
    assert a.shape == (DA,) and np.all(np.isfinite(np.asarray(a)))


# ------------------------------------------------------------------ guards

@pytest.mark.parametrize("over", [
    {"coef_source": "reward_gpi"},                                            # needs policy_index=goal
    {**ARMS["sm"], **READOUT, "reward_gpi_cands": "best"},
    {**ARMS["sm"], **READOUT, "eval_goal_pool": "dataset"},
])
def test_create_asserts_fire(over):
    with pytest.raises(AssertionError):
        _agent(**over)


# ------------------------------------------------------------------ restore

def test_checkpoint_without_the_new_fields_restores_and_acts(tmp_path):
    """A checkpoint written before eval_cand_w / eval_ref_reward existed restores into the
    reward_gpi readout, keeps its trained measure and goal head, and acts."""
    ag = _agent(seed=1, **ARMS["sm"])
    ag, _ = ag.update(_batch())
    saved = flax.serialization.to_state_dict(ag)
    for k in ("eval_cand_w", "eval_ref_reward"):
        saved.pop(k)                               # KeyError here = the field does not exist
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with open(run_dir / "params_10.pkl", "wb") as f:
        pickle.dump({"agent": saved}, f)
    restored = restore_agent(_agent(seed=2, **{**ARMS["sm"], **READOUT}), str(run_dir), 10)
    assert all(jnp.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(restored.basis.params),
                                                      jax.tree_util.tree_leaves(ag.basis.params)))
    assert all(jnp.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(restored.w_star.params),
                                                      jax.tree_util.tree_leaves(ag.w_star.params)))
    b, rew = _relabel()
    ev = restored.infer_eval_goals(b, rew)
    a = ev.sample_actions(jnp.asarray(b["observations"][0]), seed=jax.random.PRNGKey(0))
    assert a.shape == (DA,) and np.all(np.isfinite(np.asarray(a)))
