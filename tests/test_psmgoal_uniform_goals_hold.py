"""psmgoal ablations (2026-10-04): uniform hindsight goals and the held gpi latent.

A. `goal_sampling` (policy_index=goal): geometric (default, bit-identical) | uniform. Under
   uniform the measure goal of row i is drawn uniformly from the row's own s' and every later
   state of the same trajectory (`future_goal_idxs`, the fb_goals draw), capped at the
   trajectory end; the data-bootstrap row restrictions are untouched.
B. `gpi_hold` (acting=gpi, eval only): the latent chosen by `select_latent` is reused for
   gpi_hold env steps; the action is still decoded at the current state every step. The
   counter lives in the eval loop (`utils/evaluation.py`), reset at every episode start.

Run in its own process: JAX_PLATFORMS=cpu .venv/bin/python -m pytest
tests/test_psmgoal_uniform_goals_hold.py -q -p no:cacheprovider
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
from utils.datasets import Dataset
from utils.evaluation import HeldLatentActor, evaluate, gpi_hold_steps, supply_rng
from utils.flow_inversion import PREIMAGE_VALID_KEY

OB, DA, Z, CODE, N = 6, 3, 8, 4, 8

NEW_KEYS = {"goal_sampling": "geometric", "gpi_hold": 1}

DB = {"policy_index": "goal", "bootstrap_source": "data", "train_actor": False,
      "goal_random_frac": 0.0, "goal_cur_frac": 0.0}


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


# ------------------------------------------------------------------ defaults

def test_new_keys_default_in_get_config_and_yaml():
    c = get_config()
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "agent", "psmgoal.yaml")
    with open(path) as f:
        y = yaml.safe_load(f)
    for k, v in NEW_KEYS.items():
        assert k in c and c[k] == v, f"get_config()[{k!r}]"
        assert k in y and y[k] == v, f"psmgoal.yaml[{k!r}]"
    assert pg.GOAL_SAMPLINGS == ("geometric", "uniform")


def test_dataset_default_is_geometric():
    ds = _dataset()
    assert ds.goal_sampling == "geometric"


# ------------------------------------------------------------------ A. uniform goals

ENDS = (9, 19, 29, 39)


def _dataset(n=40, ends=ENDS, invalid=()):
    rng = np.random.default_rng(0)
    terminals = np.zeros((n,), np.float32)
    terminals[list(ends)] = 1.0
    valid = np.ones((n,), np.float32)
    valid[list(invalid)] = 0.0
    point = rng.standard_normal((n, DA)).astype(np.float32)
    point[list(invalid)] = 0.0
    fields = {"observations": rng.standard_normal((n, OB)).astype(np.float32),
              "next_observations": rng.standard_normal((n, OB)).astype(np.float32),
              "actions": rng.standard_normal((n, DA)).astype(np.float32),
              "terminals": terminals, "noise_preimage_point": point}
    fields[PREIMAGE_VALID_KEY] = valid
    ds = Dataset.create(**fields)
    ds.return_preimage_noise = True
    ds.preimage_point_mode = True
    return ds


def _goal_dataset(sampling, discount=0.9):
    ds = _dataset(invalid=(5, 22))
    ds.return_next_preimage = True
    ds.return_goals = True
    ds.goal_random_frac = 0.0
    ds.goal_cur_frac = 0.0
    ds.goal_discount = discount
    ds.goal_sampling = sampling
    return ds


def _goal_rows(ds, idxs, goals):
    """The dataset row each goal state came from (goals are next_observations rows)."""
    nobs = np.asarray(ds["next_observations"])
    rows = []
    for g in goals:
        hits = np.nonzero(np.all(nobs == g[None], axis=1))[0]
        assert hits.size == 1
        rows.append(int(hits[0]))
    return np.asarray(rows)


def test_default_geometric_goals_are_bit_identical():
    """goal_sampling=geometric draws exactly what the sampler drew before the key existed."""
    ds0, ds1 = _goal_dataset("geometric"), _goal_dataset("geometric")
    np.random.seed(3)
    b0 = ds0.sample(64)
    np.random.seed(3)
    idxs = ds1.get_random_idxs(64)
    from utils.datasets import hindsight_goal_idxs
    g_idx, _ = hindsight_goal_idxs(idxs, ds1.terminal_locs, ds1.size, 0.9, 0.0, cur_frac=0.0)
    np.testing.assert_array_equal(b0["goals"], np.asarray(ds1["next_observations"])[g_idx])
    np.testing.assert_array_equal(b0["noise_preimage"], np.asarray(ds1["noise_preimage_point"])[idxs])


def test_uniform_goals_lie_at_or_after_the_row_on_the_same_trajectory():
    ds = _goal_dataset("uniform")
    np.random.seed(11)
    idxs = ds.get_random_idxs(512)
    b = ds.get_subset(idxs)
    ends = np.array(ENDS)
    end_of = ends[np.searchsorted(ends, idxs)]
    rows = _goal_rows(ds, idxs, b["goals"])
    assert np.all(rows >= idxs)
    assert np.all(rows <= end_of)
    # the data-bootstrap row restrictions still hold and the next latent rides along
    assert not (set(idxs.tolist()) & {9, 19, 29, 39, 4, 21, 5, 22})
    np.testing.assert_array_equal(b["next_noise_preimage"], ds["noise_preimage_point"][idxs + 1])


def test_uniform_offsets_are_flat_and_geometric_offsets_are_not():
    """From row 0 of a 10-row trajectory the uniform goal row is one of 0..9 with equal
    probability; the geometric draw at discount 0.9 is not flat over the same range."""
    n_draw = 20000
    idxs = np.zeros((n_draw,), np.int64)          # row 0, trajectory end 9
    counts = {}
    for sampling in ("uniform", "geometric"):
        ds = _goal_dataset(sampling)
        np.random.seed(5)
        b = ds.get_subset(idxs)
        rows = _goal_rows(ds, idxs, b["goals"][:2000])   # row lookup is slow; 2000 is enough
        counts[sampling] = np.bincount(rows, minlength=10)[:10]
    exp = 2000 / 10.0
    chi_u = float(np.sum((counts["uniform"] - exp) ** 2 / exp))
    chi_g = float(np.sum((counts["geometric"] - exp) ** 2 / exp))
    # chi-square with 9 degrees of freedom: 99.9th percentile 27.9
    assert chi_u < 27.9, counts["uniform"]
    assert chi_g > 27.9, counts["geometric"]
    assert counts["uniform"].min() > 0.5 * exp and counts["uniform"].max() < 1.5 * exp
    # geometric at 0.9: the first offsets carry more mass than the last ones
    assert counts["geometric"][0] > counts["geometric"][5]


def test_uniform_goal_sampling_builds_and_trains_with_the_data_bootstrap():
    ag = _agent(**DB, goal_sampling="uniform")
    assert str(ag.config["goal_sampling"]) == "uniform"
    rng = np.random.default_rng(0)
    b = {"observations": rng.standard_normal((N, OB)).astype(np.float32),
         "next_observations": rng.standard_normal((N, OB)).astype(np.float32),
         "noise_preimage": rng.standard_normal((N, DA)).astype(np.float32),
         "next_noise_preimage": rng.standard_normal((N, DA)).astype(np.float32),
         "goals": rng.standard_normal((N, OB)).astype(np.float32),
         "index": np.arange(N).astype(np.int32)}
    _, info = ag.update(b)
    assert np.isfinite(float(info["psm_loss"]))
    # the key changes nothing inside the agent: same update as the geometric arm
    _, info_g = _agent(**DB).update(b)
    assert float(info["psm_loss"]) == float(info_g["psm_loss"])


def test_goal_sampling_assert_fires():
    with pytest.raises(AssertionError):
        _agent(**DB, goal_sampling="bogus")
    with pytest.raises(AssertionError):
        _agent(gpi_hold=0)


# ------------------------------------------------------------------ B. held latent

class _MockEnv:
    """Fixed-length episodes with changing observations; records every action."""

    def __init__(self, horizon=10):
        self.horizon = horizon
        self._t = 0
        self.actions = []
        self.episode_starts = []

    def reset(self, *, seed=None, options=None):
        self._t = 0
        self.episode_starts.append(len(self.actions))
        return self._obs(), {"success": 0.0}

    def _obs(self):
        return (0.1 * self._t) * np.ones(OB, np.float32)

    def step(self, action):
        self.actions.append(np.asarray(action))
        self._t += 1
        done = self._t >= self.horizon
        return self._obs(), 0.0, done, False, {"success": 0.0}


class _Spy:
    """Wraps a psmgoal agent, recording the latents the eval loop chooses."""

    def __init__(self, agent):
        self.agent = agent
        self.config = agent.config
        self.latents = []
        self.decoded_with = []

    def choose_latent(self, observations, seed):
        u = self.agent.choose_latent(observations, seed)
        self.latents.append(np.asarray(u))
        return u

    def act_with_latent(self, observations, u):
        self.decoded_with.append(np.asarray(u))
        return self.agent.act_with_latent(observations, u)

    def sample_actions(self, observations, seed=None, temperature=1.0):
        return self.agent.sample_actions(observations, seed=seed, temperature=temperature)


def _goal_set(agent):
    k = int(agent.config["k_goals"])
    goals = jnp.asarray(np.random.default_rng(1).standard_normal((k, OB)), jnp.float32)
    return agent.replace(eval_goals=goals, eval_w_star=agent._project(jnp.ones((Z,))))


def test_gpi_hold_steps_reads_the_config():
    assert gpi_hold_steps(_goal_set(_agent())) == 1
    assert gpi_hold_steps(_goal_set(_agent(gpi_hold=4))) == 4
    assert gpi_hold_steps(_goal_set(_agent(gpi_hold=4, acting="distill"))) == 1

    class _Plain:
        def sample_actions(self, observations, seed=None, temperature=1.0):
            return np.zeros(DA, np.float32)

    assert gpi_hold_steps(_Plain()) == 1


def test_gpi_hold_1_is_the_existing_per_step_path():
    """gpi_hold=1: evaluate() draws the key stream and calls sample_actions exactly as before."""
    ag = _goal_set(_agent())
    env = _MockEnv(horizon=6)
    evaluate(ag, env, num_eval_episodes=2, seed=3)
    actor = supply_rng(ag.sample_actions, rng=jax.random.PRNGKey(3))
    ref = _MockEnv(horizon=6)
    ref.reset(seed=3)
    for _ in range(2):
        ob, _ = ref.reset()
        done = False
        while not done:
            a = np.clip(np.array(actor(observations=ob, temperature=0)), -1, 1)
            ob, _, done, _, _ = ref.step(a)
    np.testing.assert_array_equal(np.stack(env.actions), np.stack(ref.actions))
    # and is byte-identical to an agent that never heard of the key
    spy = _Spy(ag)
    env2 = _MockEnv(horizon=6)
    evaluate(spy, env2, num_eval_episodes=2, seed=3)
    np.testing.assert_array_equal(np.stack(env.actions), np.stack(env2.actions))
    assert spy.latents == [] and spy.decoded_with == []


def test_gpi_hold_4_keeps_the_latent_for_blocks_of_four_and_resets_per_episode():
    ag = _goal_set(_agent(gpi_hold=4))
    spy = _Spy(ag)
    env = _MockEnv(horizon=10)
    evaluate(spy, env, num_eval_episodes=3, seed=3)
    assert len(env.actions) == 30
    used = np.stack(spy.decoded_with)                       # (30, DA) the latent of each step
    for ep in range(3):
        blk = used[ep * 10:(ep + 1) * 10]
        for lo in (0, 4, 8):
            block = blk[lo:lo + 4]
            assert np.all(block == block[0]), (ep, lo)       # constant within the block
        assert not np.array_equal(blk[0], blk[4])            # changes at the block boundary
        assert not np.array_equal(blk[4], blk[8])
    # 3 blocks per 10-step episode: the last block is cut at the episode end and a fresh
    # latent is chosen at the next episode's first step (reset), so 9 selections in all
    assert len(spy.latents) == 9
    assert not np.array_equal(used[9], used[10])             # episode 1's tail != episode 2's head
    # every action is the decode of the held latent at the CURRENT state
    for t, (a, u) in enumerate(zip(env.actions, used)):
        ep_t = t % 10
        obs = (0.1 * ep_t) * np.ones(OB, np.float32)
        np.testing.assert_allclose(a, np.clip(np.asarray(ag.act_with_latent(jnp.asarray(obs), jnp.asarray(u))), -1, 1),
                                   atol=1e-6)
    # the chosen latent is one of the gpi_num_u prior candidates (select_latent's argmax)
    for u in spy.latents:
        assert np.all(np.abs(u) <= float(ag.config["u_clip"]))


def test_held_actor_resets_and_counts():
    ag = _goal_set(_agent(gpi_hold=3))
    actor = HeldLatentActor(ag, 3, jax.random.PRNGKey(0))
    obs = jnp.zeros((OB,), jnp.float32)
    a0 = actor(observations=obs)
    a1 = actor(observations=obs)
    a2 = actor(observations=obs)
    np.testing.assert_array_equal(a0, a1)
    np.testing.assert_array_equal(a1, a2)
    assert actor.steps_left == 0
    actor(observations=obs)
    assert actor.steps_left == 2
    actor.reset()
    assert actor.steps_left == 0 and actor.u is None


def test_choose_and_act_match_sample_actions():
    """choose_latent + act_with_latent at the same key equals sample_actions (acting=gpi)."""
    ag = _goal_set(_agent())
    obs = jnp.asarray(np.random.default_rng(2).standard_normal((OB,)), jnp.float32)
    key = jax.random.PRNGKey(9)
    u = ag.choose_latent(obs, key)
    np.testing.assert_allclose(ag.act_with_latent(obs, u), ag.sample_actions(obs, seed=key), atol=1e-6)
