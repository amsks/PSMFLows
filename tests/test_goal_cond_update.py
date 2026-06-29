import math
import torch, gymnasium as gym
from agents.fb.flow_bc.agent import FBFlowBCAgent
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig

def _agent(**kw):
    return FBFlowBCAgent(obs_space=gym.spaces.Box(-1,1,(40,)), action_dim=5, batch_size=8, z_dim=50, L_dim=50,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4), device="cpu", **kw)

def _batch(n=8, obs=40, act=5):
    return {"observation": torch.randn(n, obs), "action": torch.rand(n, act)*2-1,
            "next": {"observation": torch.randn(n, obs), "action": torch.rand(n, act)*2-1,
                     "physics": torch.randn(n, 45), "terminated": torch.zeros(n, 1)}}

def test_v1_update_runs():
    a = _agent(goal_cond=True)
    m = a.update(_batch(), step=0)
    # FBAgent.update() converts tensor metrics to python floats for logging.
    assert math.isfinite(m["fb_loss"])


def test_v1_act_with_goal():
    a = _agent(goal_cond=True)
    a._eval_goal = torch.zeros(1, 40)
    out = a.act(torch.zeros(1, 40), torch.zeros(50))
    assert out.shape[-1] == 5
