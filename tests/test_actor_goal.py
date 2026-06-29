import torch, gymnasium as gym
from nn_models import NoiseConditionedActorArchiConfig


def test_actor_accepts_goal():
    cfg = NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2)
    act = cfg.build(gym.spaces.Box(-1, 1, (40,)), z_dim=50, action_dim=5, goal_dim=40)
    a = act(torch.randn(4, 40), torch.randn(4, 50), torch.randn(4, 5), goal=torch.randn(4, 40))
    assert a.shape == (4, 5) and a.abs().max() <= 1.0


def test_actor_goal_none_default():
    cfg = NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2)
    act = cfg.build(gym.spaces.Box(-1, 1, (40,)), z_dim=50, action_dim=5)
    a = act(torch.randn(4, 40), torch.randn(4, 50), torch.randn(4, 5))
    assert a.shape == (4, 5)


def test_fb_model_goal_cond_builds():
    from agents.fb.model import FBModel
    m = FBModel(obs_space=gym.spaces.Box(-1, 1, (40,)), action_dim=5, z_dim=50, L_dim=50, goal_dim=40, device="cpu")
    assert m.goal_dim == 40
