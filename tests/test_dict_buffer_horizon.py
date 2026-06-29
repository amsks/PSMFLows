"""Tests for DictBuffer.sample(horizon=...) — horizon=1 must be unchanged;
horizon>1 must return [B, h, ...] windows contiguous within an episode."""
import numpy as np
import pytest
import torch

from buffers.transition import DictBuffer


def _make_buffer_with_two_episodes(ep1_len: int, ep2_len: int) -> DictBuffer:
    """Build a buffer with two synthetic episodes. The 'observation' values
    are global step indices so we can verify contiguity by reading them back."""
    total = ep1_len + ep2_len
    obs = np.arange(total, dtype=np.float32).reshape(-1, 1)
    next_obs = obs + 1
    timestep = np.concatenate([np.arange(ep1_len), np.arange(ep2_len)]).astype(np.int64)
    action = np.zeros((total, 2), dtype=np.float32)
    reward = np.zeros((total,), dtype=np.float32)
    terminated = np.zeros((total,), dtype=np.float32)
    terminated[ep1_len - 1] = 1.0
    terminated[-1] = 1.0
    buf = DictBuffer(capacity=total)
    buf.extend({
        "observation": obs,
        "action": action,
        "reward": reward,
        "next": {"observation": next_obs, "terminated": terminated},
        "timestep": timestep,
    })
    return buf


def test_horizon_1_default_unchanged():
    """sample(batch_size) without horizon kwarg returns single transitions
    (same shape as before the change). Regression gate for FB."""
    buf = _make_buffer_with_two_episodes(50, 50)
    torch.manual_seed(0)
    out = buf.sample(8)
    assert out["observation"].shape == (8, 1)
    assert out["action"].shape == (8, 2)
    assert "next" in out and out["next"]["observation"].shape == (8, 1)


def test_horizon_1_explicit_kwarg_identical_to_default():
    """sample(batch_size, horizon=1) and sample(batch_size) must be byte-identical
    given the same RNG seed."""
    buf = _make_buffer_with_two_episodes(50, 50)
    torch.manual_seed(42)
    a = buf.sample(8)
    torch.manual_seed(42)
    b = buf.sample(8, horizon=1)
    for k in ("observation", "action", "reward"):
        assert torch.equal(a[k], b[k]), f"horizon=1 path diverged for key {k}"


def test_horizon_5_returns_windowed_shape():
    """horizon=5 returns [B, 5, ...] for each leaf tensor."""
    buf = _make_buffer_with_two_episodes(30, 30)
    out = buf.sample(4, horizon=5)
    assert out["observation"].shape == (4, 5, 1)
    assert out["action"].shape == (4, 5, 2)
    assert out["reward"].shape == (4, 5)
    assert out["next"]["observation"].shape == (4, 5, 1)


def test_horizon_5_windows_are_contiguous_within_episode():
    """Windows must be from a single episode (no cross-boundary bleed).
    Our synthetic buffer has obs = global step index; within a true window
    the observations increment by exactly 1."""
    buf = _make_buffer_with_two_episodes(30, 30)
    out = buf.sample(64, horizon=5)
    obs = out["observation"].cpu().numpy()
    diffs = np.diff(obs, axis=1)
    assert np.all(diffs == 1.0), f"non-contiguous window detected: diffs unique = {np.unique(diffs)}"


def test_horizon_too_long_raises():
    """If no episode is long enough for the requested horizon, sample() raises."""
    buf = _make_buffer_with_two_episodes(3, 3)
    with pytest.raises((ValueError, AssertionError)):
        buf.sample(8, horizon=10)
