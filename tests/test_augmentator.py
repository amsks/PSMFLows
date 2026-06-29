import gymnasium
import numpy as np
import torch

from nn_models import Augmentator, AugmentatorArchiConfig


def _obs_space(c=9, h=64, w=64):
    return gymnasium.spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8)


def test_augmentator_preserves_shape_dtype_device():
    aug = AugmentatorArchiConfig(pad=2).build(_obs_space())
    x = torch.rand(8, 9, 64, 64)
    out = aug(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device
    assert torch.isfinite(out).all()


def test_augmentator_actually_shifts():
    torch.manual_seed(0)
    aug = AugmentatorArchiConfig(pad=4).build(_obs_space())
    x = torch.rand(16, 9, 64, 64)
    out = aug(x)
    assert not torch.allclose(out, x), "augmentation should alter at least some samples"


def test_augmentator_rejects_non_square():
    aug = AugmentatorArchiConfig(pad=2).build(_obs_space())
    try:
        aug(torch.rand(2, 9, 64, 48))
        assert False, "expected AssertionError for non-square image"
    except AssertionError as e:
        assert "square" in str(e)


def test_augmentator_name_discriminator():
    assert AugmentatorArchiConfig().name == "random_shifts"
