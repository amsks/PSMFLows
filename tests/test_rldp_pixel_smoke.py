"""RLDP pixel (DrQ) front-end smoke.

Builds RLDPFlowBCAgent with the DrQ encoder + random-shift augmentator + RGB
normalizer (frame_stack=3 => 9 channels), feeds a *windowed* pixel batch via
DictBuffer.sample(..., horizon=h), and asserts one update step yields finite
losses. Verifies:
  - the model derives obs_dim from the encoder output_space (not raw obs shape),
  - update() applies normalize + aug + enc so the CNN front-end runs on pixels,
  - the windowed SP-loss path (future_obs slice) runs through the encoder.
"""
import math

import gymnasium as gym
import numpy as np
import torch

from agents.rldp.flow_bc.agent import RLDPFlowBCAgent
from buffers.transition import DictBuffer
from nn_models import (
    AugmentatorArchiConfig,
    DrQEncoderArchiConfig,
    NoiseConditionedActorArchiConfig,
    SimpleVectorFieldArchiConfig,
)
from normalizers import RGBNormalizerConfig

C, H, W = 9, 32, 32   # AGENT obs-space channels: frame_stack=3 RGB, square image
RGB_C = 3             # raw single-frame channel count stored in the buffer
FEATURE_DIM = 32
HORIZON = 3           # pixel-tuned horizon (state default is 5)


def _agent(**kw):
    obs_space = gym.spaces.Box(low=0, high=255, shape=(C, H, W), dtype=np.uint8)
    return RLDPFlowBCAgent(
        obs_space=obs_space, action_dim=5, batch_size=8, z_dim=32, horizon=HORIZON,
        ortho_coef=0.01, flow_steps=5, bc_coeff=0.3, lr_actor_vf=3e-4,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4),
        obs_normalizer_cfg=RGBNormalizerConfig(),
        rgb_encoder_cfg=DrQEncoderArchiConfig(feature_dim=FEATURE_DIM),
        augmentator_cfg=AugmentatorArchiConfig(pad=2),
        device="cpu", **kw)


def _pixel_buffer(n_eps=5, ep_len=40):
    total = n_eps * ep_len
    # store RAW 3-channel frames; windowed sampling must frame-stack to 9 channels
    img = lambda: np.random.randint(0, 256, (total, RGB_C, H, W), dtype=np.uint8)
    action = np.tanh(np.random.randn(total, 5)).astype(np.float32)
    reward = np.zeros((total,), dtype=np.float32)
    terminated = np.zeros((total,), dtype=bool)
    for i in range(n_eps):
        terminated[(i + 1) * ep_len - 1] = True
    timestep = np.concatenate([np.arange(ep_len) for _ in range(n_eps)]).astype(np.int64)
    buf = DictBuffer(capacity=total, frame_stack=3, obs_type="pixels")
    buf.extend({
        "observation": img(),
        "action": action,
        "reward": reward,
        "next": {"observation": img(), "terminated": terminated},
        "timestep": timestep,
    })
    return buf


def test_pixel_front_end_built_from_encoder_output_dim():
    a = _agent()
    assert type(a.model._fw_encoder).__name__ == "DrQEncoder"
    assert type(a.model._bw_encoder).__name__ == "DrQEncoder"
    assert type(a.model._augmentator).__name__ == "Augmentator"
    # obs_dim must come from the encoder output_space, not the raw (9,32,32) obs space
    assert a.model._fw_encoder.output_space.shape[0] == FEATURE_DIM


def test_pixel_update_finite():
    a = _agent()
    buf = _pixel_buffer()
    torch.manual_seed(0)
    m = a.update(buf.sample(8, horizon=HORIZON), step=0)
    for k in ["fb_loss", "sp_loss", "orth_loss", "actor_loss", "bc_flow_loss"]:
        assert math.isfinite(float(m[k])), k
