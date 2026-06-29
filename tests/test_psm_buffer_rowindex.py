"""Tests for the optional row-index passthrough on DictBuffer (PSM Task 3.1).

PSM's proto-sampler needs the buffer ROW INDEX of each sampled transition
(reference uses np.arange(N); seed = (<z,2^i> + row_index) % max_seed).

The passthrough is opt-in (constructor flag with_index=False default) so that
FB and every other caller is byte-unaffected: with the flag off, sample()
returns NO "index" key.
"""
import numpy as np
import torch

from buffers.transition import DictBuffer


def _make_buffer(total: int, with_index: bool = False) -> DictBuffer:
    """Single-episode synthetic buffer; observation = global step index so we
    can verify that a returned index maps to the right row."""
    obs = np.arange(total, dtype=np.float32).reshape(-1, 1)
    next_obs = obs + 1
    timestep = np.arange(total).astype(np.int64)
    action = np.zeros((total, 2), dtype=np.float32)
    reward = np.zeros((total,), dtype=np.float32)
    terminated = np.zeros((total,), dtype=np.float32)
    terminated[-1] = 1.0
    buf = DictBuffer(capacity=total, with_index=with_index)
    buf.extend({
        "observation": obs,
        "action": action,
        "reward": reward,
        "next": {"observation": next_obs, "terminated": terminated},
        "timestep": timestep,
    })
    return buf


def test_default_off_no_index_key():
    """Default buffer (with_index=False) must NOT attach an "index" key.
    Regression gate: FB and all other callers stay byte-identical."""
    buf = _make_buffer(100, with_index=False)
    torch.manual_seed(0)
    out = buf.sample(8)
    assert out.get("index") is None, "default sample() must not emit an 'index' key"


def test_opt_in_index_is_long_tensor_of_valid_rows():
    """With with_index=True, sample(n) returns batch["index"] as a [n] long
    tensor of valid row indices in [0, len)."""
    n = 16
    size = 100
    buf = _make_buffer(size, with_index=True)
    torch.manual_seed(0)
    out = buf.sample(n)
    assert "index" in out, "with_index=True must attach an 'index' key"
    idx = out["index"]
    assert isinstance(idx, torch.Tensor)
    assert idx.dtype == torch.long, f"index dtype should be long, got {idx.dtype}"
    assert idx.shape == (n,), f"index shape should be ({n},), got {tuple(idx.shape)}"
    assert int(idx.min()) >= 0 and int(idx.max()) < len(buf)


def test_index_matches_returned_observation_rows():
    """The returned index must actually correspond to the returned rows:
    observation[row] == buffer obs at that index (obs == global step index)."""
    buf = _make_buffer(100, with_index=True)
    torch.manual_seed(123)
    out = buf.sample(32)
    idx = out["index"].cpu().numpy()
    obs = out["observation"].cpu().numpy().reshape(-1)
    # obs values were set to the global row index, so they must match.
    assert np.array_equal(obs, idx.astype(np.float32)), (
        "returned 'index' does not line up with the returned observation rows"
    )


def test_index_does_not_perturb_other_keys():
    """Turning on with_index must not change the sampled transitions for a
    given RNG seed (the index is purely additive)."""
    torch.manual_seed(7)
    a = _make_buffer(100, with_index=False).sample(8)
    torch.manual_seed(7)
    b = _make_buffer(100, with_index=True).sample(8)
    for k in ("observation", "action", "reward"):
        assert torch.equal(a[k], b[k]), f"with_index changed sampled key {k}"
