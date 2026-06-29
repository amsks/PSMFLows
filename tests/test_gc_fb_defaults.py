import torch, gymnasium as gym
from agents.fb.flow_bc.agent import FBFlowBCAgent
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig


def _agent(**kw):
    obs_space = gym.spaces.Box(-1, 1, (40,))
    return FBFlowBCAgent(
        obs_space=obs_space, action_dim=5, batch_size=8, z_dim=50, L_dim=50,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4),
        device="cpu", **kw)


def test_defaults_off():
    a = _agent()
    assert a.goal_cond is False and a.fixed_b == "none" and a.goal_dim == 0
