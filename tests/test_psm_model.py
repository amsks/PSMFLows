import math
import gymnasium as gym
import torch
from agents.psm.model import PSMModel


def _model(**kw):
    return PSMModel(obs_space=gym.spaces.Box(-1, 1, (40,)), action_dim=5,
                    z_dim=128, max_log_seed=16, batch_size=8, device="cpu", **kw)


def test_builds_nets_and_targets():
    m = _model()
    assert m.phi is not None and m.sf_psi is not None and m.psm_psi is not None and m.actor is not None
    assert m.target_sf_psi is not None and m.target_psm_psi is not None and m.target_phi is not None


def test_sample_z_gaussian_normalized():
    m = _model(norm_z=True)
    z = m.sample_z(8, device="cpu")
    assert z.shape == (8, 128)
    assert torch.allclose(z.norm(dim=-1), torch.full((8,), math.sqrt(128)), atol=1e-4)


def test_sample_z_psm_binary_16d():
    m = _model()
    z = m.sample_z_psm(8, device="cpu")
    assert z.shape == (8, 16)
    assert set(torch.unique(z).tolist()) <= {0.0, 1.0}


def test_reward_inference_shape_and_norm():
    m = _model(norm_z=True)
    next_obs = torch.randn(32, 40)
    reward = torch.randn(32, 1)
    z = m.reward_inference(next_obs, reward)
    assert z.shape == (1, 128)
    assert torch.allclose(z.norm(dim=-1), torch.tensor([math.sqrt(128)]), atol=1e-4)
