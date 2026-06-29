"""Task 7.1 — PSMFlowBCAgent 50-step end-to-end train smoke (state + pixel).

Runs a short training loop for both the state and pixel (DrQ) front-ends and
asserts the SF/PSM/actor losses stay finite across all steps. A small
`max_log_seed` keeps the proto-table build fast.
"""

import math

import gymnasium as gym
import numpy as np
import pytest
import torch
from agents.psm.flow_bc.agent import PSMFlowBCAgent
from nn_models import (
    AugmentatorArchiConfig,
    DrQEncoderArchiConfig,
    NoiseConditionedActorArchiConfig,
    SimpleVectorFieldArchiConfig,
)
from normalizers import RGBNormalizerConfig

# ---- state ----------------------------------------------------------------


def _state_agent():
    return PSMFlowBCAgent(
        obs_space=gym.spaces.Box(-1, 1, (40,)), action_dim=5, batch_size=8, z_dim=128, max_log_seed=6,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4), device="cpu")


def _state_batch(n=8, obs=40, act=5):
    return {"observation": torch.randn(n, obs), "action": torch.rand(n, act) * 2 - 1, "index": torch.arange(n),
            "next": {"observation": torch.randn(n, obs), "terminated": torch.zeros(n, 1, dtype=torch.bool)}}


# ---- pixel (mirrors tests/test_psm_pixel_smoke.py) ------------------------

C, H, W = 9, 32, 32  # frame_stack=3 RGB, square image
FEATURE_DIM = 32


def _pixel_agent():
    obs_space = gym.spaces.Box(low=0, high=255, shape=(C, H, W), dtype=np.uint8)
    return PSMFlowBCAgent(
        obs_space=obs_space, action_dim=5, batch_size=8, z_dim=32, max_log_seed=6,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4),
        obs_normalizer_cfg=RGBNormalizerConfig(),
        rgb_encoder_cfg=DrQEncoderArchiConfig(feature_dim=FEATURE_DIM),
        augmentator_cfg=AugmentatorArchiConfig(pad=2),
        device="cpu")


def _pixel_batch(n=8):
    img = lambda: torch.randint(0, 256, (n, C, H, W), dtype=torch.uint8).float()
    return {"observation": img(), "action": torch.rand(n, 5) * 2 - 1, "index": torch.arange(n),
            "next": {"observation": img(), "terminated": torch.zeros(n, 1, dtype=torch.bool)}}


@pytest.mark.parametrize(
    "agent_fn,batch_fn",
    [(_state_agent, _state_batch), (_pixel_agent, _pixel_batch)],
    ids=["state", "pixel"],
)
def test_50_step_train_smoke(agent_fn, batch_fn):
    a = agent_fn()
    m = None
    for s in range(50):
        m = a.update(batch_fn(), step=s)
    assert math.isfinite(m["sf_loss"]), "sf_loss"
    assert math.isfinite(m["psm_loss"]), "psm_loss"
    assert math.isfinite(m["actor_loss"]), "actor_loss"
