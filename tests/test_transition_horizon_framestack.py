"""Windowed (horizon>1) sampling must frame-stack pixel observations, matching
the non-windowed path. Regression for the RLDP-pixel channel-mismatch bug:
_sample_horizon previously returned raw single frames ([B,h,3,H,W]) so the
9-channel DrQ encoder (frame_stack=3 x RGB) got 3-channel input.
"""
import numpy as np

from buffers.transition import DictBuffer


def _pixel_buf(T=60, C=3, H=16, W=16):
    buf = DictBuffer(capacity=T, frame_stack=3, obs_type="pixels")
    buf.extend({
        "observation": np.random.randint(0, 256, (T, C, H, W), dtype=np.uint8),
        "action": np.random.randn(T, 5).astype(np.float32),
        "next": {"observation": np.random.randint(0, 256, (T, C, H, W), dtype=np.uint8),
                 "terminated": np.zeros((T, 1), dtype=bool)},
        "timestep": np.arange(T).astype(np.int64),
    })
    return buf


def test_horizon1_pixels_frame_stacked():
    b = _pixel_buf().sample(8, horizon=1)
    assert tuple(b["observation"].shape) == (8, 9, 16, 16)
    assert tuple(b["next"]["observation"].shape) == (8, 9, 16, 16)


def test_horizon_window_pixels_frame_stacked():
    b = _pixel_buf().sample(8, horizon=3)
    # each of the h window steps must carry a 3-frame stack => 9 channels
    assert tuple(b["observation"].shape) == (8, 3, 9, 16, 16)
    assert tuple(b["next"]["observation"].shape) == (8, 3, 9, 16, 16)
    assert tuple(b["action"].shape) == (8, 3, 5)


def test_horizon_window_state_unchanged():
    # state buffers (frame_stack=1) must keep raw windowed shapes (byte-identical path)
    T = 60
    buf = DictBuffer(capacity=T, frame_stack=1, obs_type="state")
    buf.extend({
        "observation": np.random.randn(T, 20).astype(np.float32),
        "action": np.random.randn(T, 5).astype(np.float32),
        "next": {"observation": np.random.randn(T, 20).astype(np.float32),
                 "terminated": np.zeros((T, 1), dtype=bool)},
        "timestep": np.arange(T).astype(np.int64),
    })
    b = buf.sample(8, horizon=3)
    assert tuple(b["observation"].shape) == (8, 3, 20)
    assert tuple(b["next"]["observation"].shape) == (8, 3, 20)
