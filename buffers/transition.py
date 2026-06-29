"""
buffers/transition.py — Generic nested-dict replay buffer.

DictBuffer accepts arbitrary nested-dict data structures (e.g. the
``{"observation": ..., "next": {"observation": ..., "terminated": ...}}``
layout used by OGBench episodes) and returns the same structure from
``sample()``.

Internally all arrays are stored flat (``"next/observation"`` etc.) for
O(1) random-access; the nesting is reconstructed on sample.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
# Nested-dict helpers
# ──────────────────────────────────────────────────────────────────────────────

def _flatten(d: Dict, prefix: str = "") -> Dict[str, np.ndarray]:
    """Recursively flatten nested dict into ``{"a/b/c": array}`` form."""
    out: Dict[str, np.ndarray] = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=key))
        else:
            out[key] = np.asarray(v)
    return out


def _unflatten(flat: Dict[str, Any]) -> Dict:
    """Reconstruct nested dict from flat ``{"a/b/c": value}`` form."""
    result: Dict = {}
    for dotkey, v in flat.items():
        parts = dotkey.split("/")
        d = result
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return result


# ──────────────────────────────────────────────────────────────────────────────
# DictBuffer
# ──────────────────────────────────────────────────────────────────────────────

class DictBuffer:
    """Circular replay buffer that stores and returns arbitrary nested dicts.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to hold.
    device : str
        Device for tensors returned by ``sample()``.

    Usage
    -----
    >>> buf = DictBuffer(capacity=10_000, device="cuda")
    >>> buf.extend({"observation": obs_arr, "action": act_arr,
    ...             "next": {"observation": nobs_arr, "terminated": term_arr}})
    >>> batch = buf.sample(256)   # same nested structure, as tensors
    """

    def __init__(
        self,
        capacity: int,
        device: str = "cpu",
        frame_stack: int = 1,
        obs_type: str = "state",
        with_index: bool = False,
    ):
        self.capacity = capacity
        self.device = device
        self.frame_stack = frame_stack
        self.obs_type = obs_type
        # Opt-in (PSM proto-sampler): when True, sample() attaches the sampled
        # buffer ROW INDICES under batch["index"] ([B] long tensor). Default
        # False keeps every existing caller byte-identical (no "index" key).
        self.with_index = with_index
        self._stack = obs_type == "pixels" and frame_stack > 1
        self._size: int = 0
        self._ptr: int = 0
        self._storage: Dict[str, np.ndarray] = {}  # flat storage

    # ------------------------------------------------------------------ #
    # Adding data
    # ------------------------------------------------------------------ #

    def extend(self, data: Dict) -> None:
        """Add a batch of transitions.  ``data`` may contain nested dicts."""
        flat = _flatten(data)
        n = len(next(iter(flat.values())))

        if n > self.capacity:
            # Keep only the most recent `capacity` entries
            flat = {k: v[-self.capacity:] for k, v in flat.items()}
            n = self.capacity

        # Lazy initialisation on first extend
        if not self._storage:
            for k, v in flat.items():
                self._storage[k] = np.zeros(
                    (self.capacity, *v.shape[1:]), dtype=v.dtype
                )

        # Ring-buffer write
        end = self._ptr + n
        if end <= self.capacity:
            for k, v in flat.items():
                self._storage[k][self._ptr:end] = v
        else:
            first = self.capacity - self._ptr
            for k, v in flat.items():
                self._storage[k][self._ptr:] = v[:first]
                self._storage[k][: n - first] = v[first:]

        self._ptr = (self._ptr + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #

    def _stack_obs(self, key: str, idx: np.ndarray):
        """Stack `frame_stack` frames of `key` ending at idx, clamped at the
        episode start via stored `timestep` (td_jepa parallel.py:73-89 math).
        Returns a CHW-concatenated tensor (oldest->newest on channel axis)."""
        ts = self._storage["timestep"][idx]  # episode-relative index
        frames = []
        for k in range(self.frame_stack - 1, -1, -1):  # oldest -> newest
            back = np.minimum(k, ts)                    # never cross ep start
            frames.append(self._storage[key][idx - back])
        arr = np.concatenate(frames, axis=1)            # channel axis
        return torch.as_tensor(arr).to(self.device)

    def _stack_next_obs(self, idx: np.ndarray):
        """s' = drop oldest frame of s, append the stored next frame."""
        ts = self._storage["timestep"][idx]
        frames = []
        for k in range(self.frame_stack - 2, -1, -1):   # s minus its oldest
            back = np.minimum(k, ts)
            frames.append(self._storage["observation"][idx - back])
        frames.append(self._storage["next/observation"][idx])
        arr = np.concatenate(frames, axis=1)
        return torch.as_tensor(arr).to(self.device)

    def sample(self, batch_size: int, horizon: int = 1) -> Dict[str, Any]:
        """Sample *batch_size* transitions uniformly. Returns nested dict of tensors.

        Uses torch.randint (not np.random) so RNG consumption matches td_jepa's
        DictBuffer.sample exactly.

        horizon=1 (default) is the byte-identical legacy path — single
        transitions, shape [B, ...].

        horizon>1 returns contiguous-within-episode windows, shape [B, horizon, ...].
        A window starting at index i is valid iff timestep[i:i+horizon] is
        strictly consecutive (no episode boundary). Raises ValueError if no
        episode in the buffer is long enough to provide a horizon-h window.
        """
        assert self._size > 0, "Buffer is empty"
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if horizon > 1:
            return self._sample_horizon(batch_size, horizon)
        idx = torch.randint(0, self._size, (batch_size,)).numpy()

        if not self._stack:
            flat_tensors = {
                k: torch.as_tensor(v[idx]).to(self.device)
                for k, v in self._storage.items()
            }
            out = _unflatten(flat_tensors)
            out.pop("timestep", None)  # internal-only; not part of a batch
            if self.with_index:
                out["index"] = torch.as_tensor(idx, dtype=torch.long).to(self.device)
            return out

        if "timestep" not in self._storage:
            raise KeyError(
                "frame_stack>1 with obs_type='pixels' requires a 'timestep' "
                "array in storage (produced by data/ogbench.py pixel loader)."
            )

        out_flat = {}
        for k, v in self._storage.items():
            if k in ("observation", "next/observation", "timestep"):
                continue
            out_flat[k] = torch.as_tensor(v[idx]).to(self.device)
        out_flat["observation"] = self._stack_obs("observation", idx)
        out_flat["next/observation"] = self._stack_next_obs(idx)
        out = _unflatten(out_flat)
        if self.with_index:
            out["index"] = torch.as_tensor(idx, dtype=torch.long).to(self.device)
        return out

    def _sample_horizon(self, batch_size: int, horizon: int) -> Dict[str, Any]:
        """Return [B, horizon, ...] windows that stay within a single episode."""
        if "timestep" not in self._storage:
            raise ValueError(
                "horizon>1 sampling requires 'timestep' in storage "
                "(produced by data/ogbench.py loader)."
            )
        ts = self._storage["timestep"][: self._size]
        max_start = self._size - horizon
        if max_start < 0:
            raise ValueError(
                f"buffer size {self._size} < horizon {horizon}; no valid starts"
            )
        starts = np.arange(max_start + 1)
        ts_end = ts[starts + horizon - 1]
        ts_start = ts[starts]
        valid_mask = (ts_end - ts_start) == (horizon - 1)
        valid_starts = starts[valid_mask]
        if valid_starts.size == 0:
            raise ValueError(
                f"no episode in buffer is long enough for horizon={horizon}"
            )
        pick_idx = torch.randint(0, valid_starts.size, (batch_size,)).numpy()
        starts_sampled = valid_starts[pick_idx]
        offsets = np.arange(horizon)
        window_idx = starts_sampled[:, None] + offsets[None, :]  # [B, h]
        flat_idx = window_idx.reshape(-1)  # [B*h]
        out_flat = {}
        for k, v in self._storage.items():
            if k == "timestep":
                continue
            # pixels: observation/next-observation are frame-stacked below
            if self._stack and k in ("observation", "next/observation"):
                continue
            gathered = torch.as_tensor(v[flat_idx]).to(self.device)
            new_shape = (batch_size, horizon) + tuple(gathered.shape[1:])
            out_flat[k] = gathered.reshape(new_shape)
        if self._stack:
            # Frame-stack each of the B*h window positions (same clamped-at-
            # episode-start logic as the non-windowed pixel path), then fold the
            # horizon axis back in: [B*h, C*fs, H, W] -> [B, h, C*fs, H, W].
            obs = self._stack_obs("observation", flat_idx)
            nobs = self._stack_next_obs(flat_idx)
            out_flat["observation"] = obs.reshape((batch_size, horizon) + tuple(obs.shape[1:]))
            out_flat["next/observation"] = nobs.reshape((batch_size, horizon) + tuple(nobs.shape[1:]))
        return _unflatten(out_flat)

    # ------------------------------------------------------------------ #
    # Dunder helpers
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size >= self.capacity