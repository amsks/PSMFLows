"""Sub-task 6.3 — PSM pixel (DrQ) front-end smoke.

Builds PSMFlowBCAgent with the DrQ encoder + random-shift augmentator + RGB
normalizer (mirrors configs/domain/visual_cube_single.yaml: encoder=drq,
augmentator=random_shifts, frame_stack=3 => 9 channels), feeds a pixel batch,
and asserts one update step yields finite losses. Verifies:
  - PSMModel derives obs_dim from the encoder output_space (not raw obs shape),
  - the update applies aug + enc to obs/next_obs so the CNN front-end runs.
"""

import math

import gymnasium as gym
import numpy as np
import torch
from agents.psm.flow_bc.agent import PSMFlowBCAgent
from nn_models import (
    AugmentatorArchiConfig,
    DrQEncoderArchiConfig,
    NoiseConditionedActorArchiConfig,
    SimpleVectorFieldArchiConfig,
)
from normalizers import RGBNormalizerConfig

C, H, W = 9, 32, 32  # frame_stack=3 RGB, square image
FEATURE_DIM = 32


def _agent(**kw):
    obs_space = gym.spaces.Box(low=0, high=255, shape=(C, H, W), dtype=np.uint8)
    return PSMFlowBCAgent(
        obs_space=obs_space, action_dim=5, batch_size=8, z_dim=32, max_log_seed=6,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4),
        obs_normalizer_cfg=RGBNormalizerConfig(),
        rgb_encoder_cfg=DrQEncoderArchiConfig(feature_dim=FEATURE_DIM),
        augmentator_cfg=AugmentatorArchiConfig(pad=2),
        device="cpu", **kw)


def _pixel_batch(n=8):
    img = lambda: torch.randint(0, 256, (n, C, H, W), dtype=torch.uint8).float()
    return {"observation": img(), "action": torch.rand(n, 5) * 2 - 1, "index": torch.arange(n),
            "next": {"observation": img(), "terminated": torch.zeros(n, 1, dtype=torch.bool)}}


def test_pixel_front_end_built_from_encoder_output_dim():
    a = _agent()
    assert type(a.model._fw_encoder).__name__ == "DrQEncoder"
    assert type(a.model._augmentator).__name__ == "Augmentator"
    # obs_dim must come from the encoder output_space, not the raw (9,32,32) obs space
    assert a.model._fw_encoder.output_space.shape[0] == FEATURE_DIM


def test_pixel_update_finite():
    a = _agent()
    m = a.update(_pixel_batch(), step=0)
    for k in ["psm_loss", "sf_loss", "actor_loss", "bc_flow_loss"]:
        assert math.isfinite(m[k]), k
