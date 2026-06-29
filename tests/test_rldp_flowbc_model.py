"""Smoke tests for RLDPFlowBCModel (subclass of RLDPModel that adds _actor_vf)."""
import gymnasium
import numpy as np
import torch

from agents.rldp.flow_bc.model import RLDPFlowBCModel


def _obs_space(dim: int):
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def test_rldp_flowbc_model_has_actor_vf_and_predictor():
    model = RLDPFlowBCModel(obs_space=_obs_space(20), action_dim=4)
    for attr in ("_actor_vf", "_predictor", "_backward_map", "_forward_map"):
        assert hasattr(model, attr), f"missing submodule {attr}"


def test_rldp_flowbc_model_act_returns_actions():
    model = RLDPFlowBCModel(obs_space=_obs_space(20), action_dim=4)
    obs = torch.randn(8, 20)
    z = torch.randn(8, model.z_dim)
    actions = model.act(obs, z)
    assert actions.shape == (8, 4)
    assert torch.isfinite(actions).all()
