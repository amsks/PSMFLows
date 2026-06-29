import math
import gymnasium as gym
import torch
from agents.psm.flow_bc.agent import PSMFlowBCAgent
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig


def _agent(**kw):
    return PSMFlowBCAgent(
        obs_space=gym.spaces.Box(-1, 1, (40,)), action_dim=5, batch_size=8, z_dim=128, max_log_seed=16,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4),
        device="cpu", **kw)


def _batch(n=8, obs=40, act=5):
    return {"observation": torch.randn(n, obs), "action": torch.rand(n, act) * 2 - 1, "index": torch.arange(n),
            "next": {"observation": torch.randn(n, obs), "terminated": torch.zeros(n, 1, dtype=torch.bool)}}


def test_flowbc_update_and_act():
    a = _agent()
    m = a.update(_batch(), step=0)
    assert math.isfinite(m["sf_loss"]) and math.isfinite(m["actor_loss"])
    out = a.act(torch.zeros(1, 40), torch.zeros(128))
    assert out.shape[-1] == 5
