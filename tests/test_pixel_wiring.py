from omegaconf import OmegaConf

from train import build_pixel_cfgs  # helper added in this task


def test_state_cfg_returns_identity_defaults():
    cfg = OmegaConf.create({})  # no pixel blocks
    norm, rgb, aug = build_pixel_cfgs(cfg)
    assert type(norm).__name__ == "IdentityNormalizerConfig"
    assert type(rgb).__name__ == "IdentityNNConfig"
    assert type(aug).__name__ == "IdentityNNConfig"


def test_pixel_cfg_selects_drq_rgbnorm_augmentator():
    cfg = OmegaConf.create(
        {
            "obs_normalizer": {"name": "RGBNormalizerConfig"},
            "rgb_encoder": {"name": "drq", "feature_dim": 256},
            "augmentator": {"name": "random_shifts", "pad": 2},
        }
    )
    norm, rgb, aug = build_pixel_cfgs(cfg)
    assert type(norm).__name__ == "RGBNorm" or type(norm).__name__ == "RGBNormalizerConfig"
    assert type(rgb).__name__ == "DrQEncoderArchiConfig"
    assert rgb.feature_dim == 256
    assert type(aug).__name__ == "AugmentatorArchiConfig"
    assert aug.pad == 2


def _tiny_box(shape, dtype):
    import numpy as np
    import gymnasium
    return gymnasium.spaces.Box(low=0, high=1, shape=shape, dtype=dtype)


def test_state_agent_constructs_with_identity_cfgs():
    """Regression: make_agent passes rgb_encoder_cfg/augmentator_cfg into the
    agent; FBAgent must accept+forward them (state path = Identity)."""
    import numpy as np
    from agents.fb.agent import FBAgent
    from nn_models import IdentityNNConfig
    from normalizers import IdentityNormalizerConfig

    obs_space = _tiny_box((4,), np.float32)
    agent = FBAgent(
        obs_space=obs_space,
        action_dim=2,
        z_dim=8,
        L_dim=8,
        obs_normalizer_cfg=IdentityNormalizerConfig(),
        rgb_encoder_cfg=IdentityNNConfig(),
        augmentator_cfg=IdentityNNConfig(),
        device="cpu",
    )
    assert agent.model is not None


def test_pixel_flowbc_agent_constructs_with_drq():
    """FBFlowBCAgent must thread rgb_encoder_cfg/augmentator_cfg through
    **fb_kwargs -> FBAgent -> model (pixel path = DrQ/Augmentator)."""
    import numpy as np
    from agents.fb.flow_bc.agent import FBFlowBCAgent
    from nn_models import (
        AugmentatorArchiConfig,
        DrQEncoderArchiConfig,
        NoiseConditionedActorArchiConfig,
        SimpleVectorFieldArchiConfig,
    )
    from normalizers import RGBNormalizerConfig

    obs_space = _tiny_box((9, 16, 16), np.uint8)
    agent = FBFlowBCAgent(
        obs_space=obs_space,
        action_dim=2,
        z_dim=8,
        L_dim=8,
        actor_cfg=NoiseConditionedActorArchiConfig(),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(),
        obs_normalizer_cfg=RGBNormalizerConfig(),
        rgb_encoder_cfg=DrQEncoderArchiConfig(feature_dim=32),
        augmentator_cfg=AugmentatorArchiConfig(pad=2),
        device="cpu",
    )
    assert type(agent.model._fw_encoder).__name__ == "DrQEncoder"
    assert type(agent.model._augmentator).__name__ == "Augmentator"
