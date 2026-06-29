"""Smoke tests for RLDPModel (subclass of FBModel that adds _predictor)."""
import gymnasium
import numpy as np
import torch

from agents.rldp.model import RLDPModel
from nn_models import VForwardArchiConfig


def _obs_space(dim: int):
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def test_rldp_model_has_predictor_submodule():
    model = RLDPModel(obs_space=_obs_space(20), action_dim=4)
    assert hasattr(model, "_predictor"), "RLDPModel must register _predictor"
    assert isinstance(model._predictor, torch.nn.Module)
    assert model._predictor.num_parallel == 2


def test_rldp_model_inherits_fb_submodules():
    model = RLDPModel(obs_space=_obs_space(20), action_dim=4)
    for attr in ("_backward_map", "_forward_map", "_actor", "_bw_encoder", "_fw_encoder", "_left_encoder"):
        assert hasattr(model, attr), f"missing inherited submodule {attr}"


def test_rldp_model_predictor_forward():
    """Predictor maps (B, z_dim) + (B, action_dim) -> (num_parallel, B, z_dim)."""
    model = RLDPModel(obs_space=_obs_space(20), action_dim=4)
    B, z_dim = 8, model.z_dim
    z = torch.randn(B, z_dim)
    action = torch.randn(B, 4)
    out = model._predictor(z, action)
    assert out.shape == (2, B, z_dim), out.shape


def test_rldp_model_custom_predictor_cfg():
    """Override predictor config kwarg."""
    cfg = VForwardArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2, num_parallel=1)
    model = RLDPModel(obs_space=_obs_space(20), action_dim=4, predictor_cfg=cfg)
    assert model._predictor.hidden_dim == 64
    assert model._predictor.num_parallel == 1
