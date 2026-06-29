import math
import gymnasium as gym
import torch
from agents.psm.agent import PSMAgent


def _agent(**kw):
    return PSMAgent(obs_space=gym.spaces.Box(-1, 1, (40,)), action_dim=5, batch_size=8,
                    z_dim=128, max_log_seed=16, device="cpu", **kw)


def _batch(n=8, obs=40, act=5):
    return {"observation": torch.randn(n, obs), "action": torch.rand(n, act) * 2 - 1,
            "index": torch.arange(n),
            "next": {"observation": torch.randn(n, obs), "terminated": torch.zeros(n, 1, dtype=torch.bool)}}


def test_one_update_finite():
    a = _agent()
    m = a.update(_batch(), step=0)
    for k in ["psm_loss", "sf_loss", "actor_loss"]:
        assert math.isfinite(m[k]), k
