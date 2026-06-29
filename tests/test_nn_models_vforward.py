"""Unit tests for the restored VForwardArchiConfig + VForwardMap (RLDP predictor head)."""
import gymnasium
import numpy as np
import torch

from nn_models import VForwardArchiConfig, VForwardMap


def _obs_space(dim: int):
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def test_vforward_archi_config_defaults():
    cfg = VForwardArchiConfig()
    assert cfg.hidden_dim == 1024
    assert cfg.hidden_layers == 1
    assert cfg.embedding_layers == 2
    assert cfg.num_parallel == 2


def test_vforward_config_build_returns_module():
    cfg = VForwardArchiConfig(num_parallel=2)
    m = cfg.build(_obs_space(10), z_dim=4)
    assert isinstance(m, VForwardMap)


def test_vforward_map_forward_shape_default_output():
    cfg = VForwardArchiConfig(hidden_dim=64, hidden_layers=1, embedding_layers=2, num_parallel=2)
    m = VForwardMap(_obs_space(10), z_dim=4, cfg=cfg)
    obs = torch.randn(8, 10)
    z = torch.randn(8, 4)
    out = m(obs, z)
    assert out.shape == (2, 8, 4), out.shape


def test_vforward_map_forward_shape_explicit_output_dim():
    cfg = VForwardArchiConfig(hidden_dim=64, hidden_layers=1, embedding_layers=2, num_parallel=1)
    m = VForwardMap(_obs_space(10), z_dim=4, output_dim=7, cfg=cfg)
    obs = torch.randn(8, 10)
    z = torch.randn(8, 4)
    out = m(obs, z)
    assert out.shape[-1] == 7, out.shape


def test_vforward_map_backward_pass_runs():
    cfg = VForwardArchiConfig(hidden_dim=32, hidden_layers=1, embedding_layers=2, num_parallel=2)
    m = VForwardMap(_obs_space(5), z_dim=3, cfg=cfg)
    obs = torch.randn(4, 5)
    z = torch.randn(4, 3)
    out = m(obs, z)
    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
