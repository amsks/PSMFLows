import torch
import types
from agents.tdmpc2.agent import TDMPC2Agent, GOAL_DIM, CUBE_SLICE


def _agent(obs_dim=19, act_dim=5):
    obs_space = types.SimpleNamespace(shape=(obs_dim,))
    return TDMPC2Agent(obs_space, act_dim, device="cpu", horizon=3)


def test_reward_zero_at_goal_minus_one_far():
    a = _agent()
    B, H, P = 4, 3, 40
    phys = torch.zeros(B, H, P)
    phys[..., CUBE_SLICE] = 0.0
    goal = torch.zeros(B, GOAL_DIM)            # cube already at goal
    r = a._reward(phys, goal)
    assert r.shape == (B, H, 1)
    assert torch.all(r == 0.0)
    phys[..., CUBE_SLICE] = 1.0                # far
    assert torch.all(a._reward(phys, torch.zeros(B, GOAL_DIM)) == -1.0)


def test_fold_dims():
    a = _agent(obs_dim=19)
    obs2d = torch.randn(8, 19); goal = torch.randn(8, GOAL_DIM)
    assert a._fold(obs2d, goal).shape == (8, 19 + GOAL_DIM)
    obs3d = torch.randn(4, 8, 19)              # [T,B,obs]
    assert a._fold(obs3d, goal).shape == (4, 8, 19 + GOAL_DIM)


def test_sample_goals_shape():
    a = _agent()
    phys = torch.randn(6, 3, 40)
    assert a._sample_goals(phys).shape == (6, GOAL_DIM)
