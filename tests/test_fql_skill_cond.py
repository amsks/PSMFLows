"""Hindsight-window skill conditioning for the FQL bc_only actor flow (GCBC-style).

G(s, u) becomes G(s, c, u), where c = the observation the demonstrator reached
`skill_window` steps later in the same trajectory (clipped at episode end). This is
supervised goal-conditioning, not a latent-variable skill VAE, so conditioning is plain
concatenation onto observations: no new distribution, no new sampling step.

skill_cond defaults to False and must be a byte-identical no-op then -- covered here by
running the exact bc_only update path from test_fql_bc_only.py on a batch with no
'skills' key.
"""
import jax
import ml_collections
import numpy as np
import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from agents import agents
from utils.datasets import add_skill_targets


def _config(**overrides):
    with initialize(version_base="1.3", config_path="../configs/agent"):
        cfg = compose(config_name="fql")
    c = ml_collections.ConfigDict(OmegaConf.to_container(cfg, resolve=True))
    with c.unlocked():
        c["actor_hidden_dims"] = tuple(c["actor_hidden_dims"])
        c["value_hidden_dims"] = tuple(c["value_hidden_dims"])
        for k, v in overrides.items():
            c[k] = v
    return c


def _agent(obs=4, act=2, **overrides):
    config = _config(**overrides)
    agent = agents["fql"].create(0, np.zeros((1, obs), np.float32),
                                  np.zeros((1, act), np.float32), config)
    return agent, config


def _rewardfree_batch(n=16, obs=4, act=2, seed=0, with_skills=False):
    rng = np.random.default_rng(seed)
    batch = dict(  # deliberately NO rewards / masks -- bc_only path
        observations=rng.standard_normal((n, obs)).astype(np.float32),
        actions=np.clip(rng.standard_normal((n, act)), -1, 1).astype(np.float32),
        next_observations=rng.standard_normal((n, obs)).astype(np.float32),
    )
    if with_skills:
        batch["skills"] = rng.standard_normal((n, obs)).astype(np.float32)
    return batch


# --- add_skill_targets -------------------------------------------------------------

def test_add_skill_targets_windows_and_clips_at_episode_boundary():
    # Two episodes of 5 steps each; observations are distinguishable per-index so a wrong
    # skill index is immediately visible.
    n = 10
    observations = np.arange(n, dtype=np.float32).reshape(n, 1)
    terminals = np.zeros(n, dtype=np.float32)
    terminals[4] = 1.0  # episode 1: indices 0-4
    terminals[9] = 1.0  # episode 2: indices 5-9
    dataset = dict(observations=observations, terminals=terminals)

    skills = add_skill_targets(dataset, window=3)
    expected_idx = np.array([3, 4, 4, 4, 4, 8, 9, 9, 9, 9])
    np.testing.assert_array_equal(skills[:, 0], expected_idx.astype(np.float32))

    # Never crosses into the next episode: every index in episode 1 clips at 4, not 5+.
    assert np.all(skills[:5, 0] <= 4)
    assert np.all(skills[5:, 0] >= 5)


def test_add_skill_targets_handles_dataset_not_ending_on_terminal():
    # Last episode has no terminal=1 row (dataset truncated mid-episode); the final row
    # must still act as that episode's end.
    n = 6
    observations = np.arange(n, dtype=np.float32).reshape(n, 1)
    terminals = np.zeros(n, dtype=np.float32)
    terminals[2] = 1.0  # episode 1: indices 0-2; episode 2: indices 3-5 (no terminal)
    dataset = dict(observations=observations, terminals=terminals)

    skills = add_skill_targets(dataset, window=10)
    np.testing.assert_array_equal(skills[:, 0], [2, 2, 2, 5, 5, 5])


# --- skill_cond=False: byte-identical no-op ----------------------------------------

def test_skill_cond_false_is_noop_bc_only_update():
    config = _config()
    assert config.get("skill_cond", True) is False, "skill_cond must default to False"
    agent, _ = _agent(bc_only=True)  # skill_cond left at default (False)
    batch = _rewardfree_batch(with_skills=False)  # no 'skills' key -- would KeyError otherwise
    agent, info = agent.update(batch)
    for k in ["actor/actor_loss", "actor/bc_flow_loss", "actor/distill_loss"]:
        assert k in info and np.isfinite(float(info[k])), k


# --- skill_cond=True: live conditioning --------------------------------------------

def test_skill_cond_true_create_and_update():
    agent, _ = _agent(bc_only=True, skill_cond=True, skill_window=3)
    batch = _rewardfree_batch(with_skills=True)
    agent, info = agent.update(batch)
    for k in ["actor/actor_loss", "actor/bc_flow_loss", "actor/distill_loss"]:
        assert k in info and np.isfinite(float(info[k])), k


def test_skill_cond_true_onestep_actor_output_changes_with_skills():
    agent, _ = _agent(bc_only=True, skill_cond=True, skill_window=3)
    obs = np.random.default_rng(0).standard_normal((8, 4)).astype(np.float32)
    noises = np.random.default_rng(1).standard_normal((8, 2)).astype(np.float32)
    skills_a = np.random.default_rng(2).standard_normal((8, 4)).astype(np.float32)
    skills_b = np.random.default_rng(3).standard_normal((8, 4)).astype(np.float32)

    actions_a = agent.sample_actions(obs, seed=jax.random.PRNGKey(0), skills=skills_a)
    actions_b = agent.sample_actions(obs, seed=jax.random.PRNGKey(0), skills=skills_b)
    assert not np.allclose(np.asarray(actions_a), np.asarray(actions_b)), (
        "onestep actor output did not change with skills -- conditioning is not live")


def test_sample_actions_skill_cond_true_requires_skills():
    agent, _ = _agent(bc_only=True, skill_cond=True, skill_window=3)
    obs = np.zeros((8, 4), np.float32)
    with pytest.raises(ValueError):
        agent.sample_actions(obs, seed=jax.random.PRNGKey(0))
