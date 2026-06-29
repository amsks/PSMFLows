import math
import torch, gymnasium as gym
from agents.fb.flow_bc.agent import FBFlowBCAgent
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig
import pytest

CFGS = [dict(), dict(L_dim=256), dict(goal_cond=True), dict(fixed_b="cube_xyz")]

def _batch(n=8, obs=40, act=5):
    return {"observation": torch.randn(n, obs), "action": torch.rand(n, act)*2-1,
            "next": {"observation": torch.randn(n, obs), "action": torch.rand(n, act)*2-1,
                     "physics": torch.randn(n, 45), "terminated": torch.zeros(n, 1)}, }

@pytest.mark.parametrize("extra", CFGS)
def test_50_steps(extra):
    extra = dict(extra)              # don't mutate the shared CFGS dict across params
    L = extra.pop("L_dim", 50)
    a = FBFlowBCAgent(obs_space=gym.spaces.Box(-1,1,(40,)), action_dim=5, batch_size=8, z_dim=50, L_dim=L,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4), device="cpu", **extra)
    m = None
    for s in range(50):
        m = a.update(_batch(), step=s)
    assert math.isfinite(m["fb_loss"])
