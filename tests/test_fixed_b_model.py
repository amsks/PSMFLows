import torch, gymnasium as gym
from agents.fb.model import FBModel
from agents.fb.fixed_backward import CUBE_OBS_SLICE


def test_fixed_b_model_zdim3():
    obs_space = gym.spaces.Box(-1, 1, (40,))
    m = FBModel(obs_space=obs_space, action_dim=5, z_dim=50, L_dim=50, fixed_b="cube_xyz", device="cpu")
    assert m.z_dim == 3
    obs = torch.zeros(2, 40)
    obs[:, CUBE_OBS_SLICE] = torch.tensor([0.1, 0.2, 0.3])
    b = m.backward_map(obs)
    assert b.shape == (2, 3)
