import torch, gymnasium as gym
from nn_models import ForwardArchiConfig


def test_forwardmap_accepts_goal():
    cfg = ForwardArchiConfig(hidden_dim=64, hidden_layers=1, embedding_layers=2, num_parallel=2)
    fm = cfg.build(gym.spaces.Box(-1, 1, (16,)), z_dim=50, action_dim=5, goal_dim=16)
    out = fm(torch.randn(4, 16), torch.randn(4, 50), torch.randn(4, 5), goal=torch.randn(4, 16))
    assert out.shape == (2, 4, 50)


def test_forwardmap_goal_none_default():
    cfg = ForwardArchiConfig(hidden_dim=64, hidden_layers=1, embedding_layers=2, num_parallel=2)
    fm = cfg.build(gym.spaces.Box(-1, 1, (16,)), z_dim=50, action_dim=5)  # goal_dim defaults 0
    out = fm(torch.randn(4, 16), torch.randn(4, 50), torch.randn(4, 5))
    assert out.shape == (2, 4, 50)
