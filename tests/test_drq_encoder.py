import gymnasium
import numpy as np
import torch

from nn_models import DrQEncoder, DrQEncoderArchiConfig


def _obs_space(c=9, h=64, w=64):
    return gymnasium.spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8)


def test_drq_encoder_forward_shape_with_feature_dim():
    enc = DrQEncoderArchiConfig(feature_dim=256).build(_obs_space())
    out = enc(torch.zeros(4, 9, 64, 64))
    assert out.shape == (4, 256)
    assert enc.output_space.shape == (256,)


def test_drq_encoder_gradients_flow():
    enc = DrQEncoderArchiConfig(feature_dim=256).build(_obs_space())
    x = torch.rand(2, 9, 64, 64, requires_grad=True)
    enc(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_drq_encoder_rejects_non_3d_obs_space():
    bad = gymnasium.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
    try:
        DrQEncoderArchiConfig(feature_dim=32).build(bad)
        assert False, "expected AssertionError for non-3D obs_space"
    except AssertionError as e:
        assert "3D shape" in str(e)


def test_drq_encoder_name_discriminator():
    assert DrQEncoderArchiConfig().name == "drq"
