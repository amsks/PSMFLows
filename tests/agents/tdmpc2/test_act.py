import torch, types
from agents.tdmpc2.agent import TDMPC2Agent, GOAL_DIM


def _agent(obs_dim=19, act_dim=5):
    return TDMPC2Agent(types.SimpleNamespace(shape=(obs_dim,)), act_dim, device="cpu", horizon=3)


def test_act_shape_and_range():
    a = _agent(); a.reset()
    obs = torch.randn(1, 19); goal = torch.randn(GOAL_DIM)
    act = a.act(obs, goal, eval_mode=True)
    assert act.shape == (1, 5)
    assert torch.all(act.abs() <= 1.0 + 1e-5)


def test_reset_clears_t0_and_prev_mean():
    a = _agent()
    a.core._prev_mean.add_(1.0); a._t0 = False
    a.reset()
    assert a._t0 is True
    assert torch.all(a.core._prev_mean == 0.0)
