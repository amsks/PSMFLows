"""Sub-task 6.2 — PSM reward-inference must match FB's eval protocol.

evals/ogbench.py:_infer_z calls
    self.agent.model.reward_inference(next_obs=..., reward=...)
with keyword args. PSMModel.reward_inference(next_obs, reward) must accept that
and return a z usable as the policy task vector. (norm_z) => ||z|| == sqrt(z_dim).
"""

import math

import gymnasium as gym
import torch
from agents.psm.flow_bc.agent import PSMFlowBCAgent
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig


def _agent(**kw):
    return PSMFlowBCAgent(
        obs_space=gym.spaces.Box(-1, 1, (40,)), action_dim=5, batch_size=8,
        z_dim=128, max_log_seed=6,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4),
        device="cpu", **kw)


def test_reward_inference_keyword_call_matches_fb_protocol():
    a = _agent()
    n = 32
    next_obs = torch.randn(n, 40)
    reward = torch.randn(n, 1)
    # exact call shape used by evals/ogbench.py:_infer_z
    z = a.model.reward_inference(next_obs=next_obs, reward=reward)
    assert z.shape == (1, 128)
    assert torch.isfinite(z).all()
    # norm_z=True => ||z|| == sqrt(z_dim)
    assert math.isclose(float(z.norm()), math.sqrt(128), rel_tol=1e-4)


def test_agent_infer_z_works():
    a = _agent()
    n = 32
    next_obs = torch.randn(n, 40)
    reward = torch.randn(n, 1)
    z = a.infer_z(next_obs, reward)
    assert z.shape == (1, 128)
    assert torch.isfinite(z).all()
