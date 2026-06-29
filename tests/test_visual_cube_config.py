import gymnasium
import numpy as np
from hydra import compose, initialize

from train import build_pixel_cfgs


def test_visual_cube_config_composes_pixel_blocks():
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["domain=visual_cube_single"])
    assert cfg.obs_type == "pixels"
    assert cfg.frame_stack == 3
    assert cfg.agent == "fb_flowbc"
    assert cfg.obs_normalizer.name == "RGBNormalizerConfig"
    assert cfg.rgb_encoder.name == "drq"
    assert cfg.rgb_encoder.feature_dim == 256
    assert cfg.augmentator.name == "random_shifts"
    assert cfg.augmentator.pad == 2
    assert cfg.L_dim == 256
    assert cfg.z_dim == 50
    assert cfg.batch_size == 256


def test_visual_cube_builds_real_pixel_modules():
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["domain=visual_cube_single"])
    norm, rgb, aug = build_pixel_cfgs(cfg)
    obs_space = gymnasium.spaces.Box(
        low=0, high=255, shape=(9, 64, 64), dtype=np.uint8
    )
    rgb_mod = rgb.build(obs_space)
    aug_mod = aug.build(obs_space)
    norm_mod = norm.build(obs_space)
    assert type(rgb_mod).__name__ == "DrQEncoder"
    assert type(aug_mod).__name__ == "Augmentator"
    assert type(norm_mod).__name__ == "RGBNorm"
    assert rgb_mod.output_space.shape == (256,)


def test_state_domain_still_identity():
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["domain=cube_single"])
    norm, rgb, aug = build_pixel_cfgs(cfg)
    assert type(rgb).__name__ == "IdentityNNConfig"
    assert type(aug).__name__ == "IdentityNNConfig"
    assert type(norm).__name__ == "IdentityNormalizerConfig"
