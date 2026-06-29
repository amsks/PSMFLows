"""Unit tests for FBAgent.eval_context (Task 3.1).

The agent unit test uses the default (Identity) obs normalizer and a fake env, so
normalization is identity here. The production running-stats normalization (and the
evaluator's _goal_observation/_relabel_subsample, which need a real buffer + mujoco
relabel env) are NOT covered here — they are verified by the Task 4.2 2k-step GPU run.
"""

import types

import gymnasium as gym
import torch

from agents.fb.flow_bc.agent import FBFlowBCAgent
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig


def _agent(**kw):
    return FBFlowBCAgent(
        obs_space=gym.spaces.Box(-1, 1, (40,)), action_dim=5, batch_size=8, z_dim=50, L_dim=50,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4), device="cpu", **kw)


def _fake_env():
    e = types.SimpleNamespace()
    e.unwrapped = types.SimpleNamespace(cur_task_info={"goal_xyzs": [[0.3, 0.0, 0.05]]})
    return e


def test_fixed_b_eval_context_zdim3():
    a = _agent(fixed_b="cube_xyz")             # z_dim collapses to 3 for fixed-B
    z, m = a.eval_context(env=_fake_env(), domain="cube-single-play-v0", task="t1")
    assert z.shape == (3,) and a._eval_goal is None and m["goal_source"] == 0.0


def test_v1_eval_context_stashes_normalized_goal():
    a = _agent(goal_cond=True)
    g = torch.randn(1, 40)
    z, m = a.eval_context(env=_fake_env(), domain="cube-single-play-v0", task="t1", goal_obs=g)
    assert z.shape == (50,) and a._eval_goal.shape == (1, 40) and m["goal_source"] == 1.0


def test_v1_eval_context_requires_goal_obs():
    import pytest
    a = _agent(goal_cond=True)
    with pytest.raises(ValueError):
        a.eval_context(env=_fake_env(), domain="cube-single-play-v0", task="t1")
