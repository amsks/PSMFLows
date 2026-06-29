import torch, gymnasium as gym
from agents.fb.flow_bc.agent import FBFlowBCAgent
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig


def _batch(n=8, obs=40, act=5):
    return {
        "observation": torch.randn(n, obs),
        "action": torch.rand(n, act) * 2 - 1,
        "next": {"observation": torch.randn(n, obs),
                 "action": torch.rand(n, act) * 2 - 1,
                 "physics": torch.randn(n, 21),
                 "terminated": torch.zeros(n, 1)},
    }


def test_fixed_b_update_runs():
    a = FBFlowBCAgent(
        obs_space=gym.spaces.Box(-1, 1, (40,)), action_dim=5, batch_size=8, z_dim=50, L_dim=50,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4),
        device="cpu", fixed_b="cube_xyz")
    assert a.backward_optimizer is None
    m = a.update(_batch(), step=0)
    assert torch.isfinite(torch.tensor(m["fb_loss"])).all()
