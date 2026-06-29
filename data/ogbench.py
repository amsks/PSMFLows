"""
data/ogbench.py — Offline dataset loader for OGBench.

Episode .npz files are expected at:
    <data_path>/<domain>/buffer/ep*.npz

Each file stores one episode with keys:
    observation  [T, obs_dim]    float32
    action       [T, act_dim]    float32
    physics      [T, phys_dim]   float32  (used for reward relabelling at eval)
    discount     [T]             float32  (0 at terminal transitions)

Episodes of T timesteps yield T-1 transitions:
    (obs[t], action[t]) -> (obs[t+1], physics[t+1], terminated)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from buffers.transition import DictBuffer


def load_transitions(
    npz_files: List[Union[str, Path]], obs_type: str = "state"
) -> Dict:
    """Load episode .npz files into a flat transition dict.

    State (obs_type="state") path is unchanged. Pixel (obs_type="pixels")
    path reads the "pixels" key, stores per-frame CHW uint8, and emits an
    episode-relative "timestep" for frame-stack clamping.
    """
    obs_list, act_list, next_obs_list, term_list, next_phys_list = [], [], [], [], []
    next_act_list = []
    ts_list = []
    has_physics = True

    for f in npz_files:
        ep = np.load(f)
        act = ep["action"].astype(np.float32)

        if obs_type == "pixels":
            if "pixels" not in ep:
                raise KeyError(
                    f"obs_type='pixels' but episode {f} has no 'pixels' key "
                    f"(keys: {list(ep.keys())})"
                )
            # [T, H, W, C] uint8 -> [T, C, H, W] uint8 (CHW per frame)
            pix = np.moveaxis(ep["pixels"], -1, 1)
            obs_full = np.ascontiguousarray(pix)
        elif obs_type == "state":
            obs_full = ep["observation"].astype(np.float32)
        else:
            raise ValueError(f"Unsupported obs_type {obs_type!r}")

        # EXORL-style shift (see module docstring / td_jepa base.py): action[t]
        # is the action that LED to obs[t], action[0]=0, so undo via act[1:].
        obs_list.append(obs_full[:-1])
        act_list.append(act[1:])
        next_obs_list.append(obs_full[1:])
        # SARSA next-action a' (action taken at next_obs) = act shifted one more; the
        # episode's last transition has no a', pad with its own action. Aligned with
        # act[1:] (len T-1). Used only by one-step FB (agent reads batch["next"]["action"]).
        next_act_list.append(np.concatenate([act[2:], act[-1:]], axis=0))
        n_trans = obs_full.shape[0] - 1
        term_list.append(np.zeros((n_trans, 1), dtype=bool))
        ts_list.append(np.arange(n_trans, dtype=np.int32))

        if has_physics and "physics" in ep:
            phys = ep["physics"].astype(np.float32)
            next_phys_list.append(phys[1:])
        else:
            has_physics = False

    storage: Dict = {
        "observation": np.concatenate(obs_list, axis=0),
        "action": np.concatenate(act_list, axis=0),
        "next": {
            "observation": np.concatenate(next_obs_list, axis=0),
            "terminated": np.concatenate(term_list, axis=0),
            "action": np.concatenate(next_act_list, axis=0),
        },
    }
    # timestep is always populated: frame_stack uses it for episode-boundary clamping
    # (pixel path), and DictBuffer.sample(horizon=h) uses it to enforce
    # contiguous-within-episode windows (RLDP). Cheap (int64 per transition).
    if ts_list:
        storage["timestep"] = np.concatenate(ts_list, axis=0)
    if has_physics and next_phys_list:
        storage["next"]["physics"] = np.concatenate(next_phys_list, axis=0)

    return storage


def load_ogbench_dataset(
    domain: str,
    data_path: Union[str, Path] = "datasets",
    load_n_episodes: int = 1000,
    device: str = "cpu",
    n_transitions: int | None = None,
    obs_type: str = "state",
    frame_stack: int = 1,
    with_index: bool = False,
) -> DictBuffer:
    """Load offline OGBench data into a DictBuffer.

    Parameters
    ----------
    domain : e.g. "antmaze-medium-navigate-v0"
    data_path : root dir; episodes are at <data_path>/<domain>/buffer/*.npz
    load_n_episodes : max number of episodes to load
    device : device for sampled tensors
    n_transitions : if set, truncate the buffer to this many transitions
    obs_type : "state" (default) or "pixels"
    frame_stack : number of frames to stack on sample (pixels only; 1 = off)
    with_index : opt-in (PSM proto-sampler) — when True the built DictBuffer
        emits the sampled GLOBAL buffer row indices under batch["index"].
        Default False keeps every other caller (FB, etc.) byte-identical.
    """
    buf_dir = Path(data_path) / domain / "buffer"
    files = sorted(buf_dir.glob("*.npz"))[:load_n_episodes]
    if not files:
        raise FileNotFoundError(
            f"No .npz episode files found at {buf_dir}. "
            "Download the OGBench dataset or check your data_path."
        )

    storage = load_transitions(files, obs_type=obs_type)

    if n_transitions is not None:
        for k, v in list(storage.items()):
            if isinstance(v, dict):
                storage[k] = {kk: vv[:n_transitions] for kk, vv in v.items()}
            else:
                storage[k] = v[:n_transitions]

    n = len(storage["observation"])
    buffer = DictBuffer(
        capacity=n, device=device, frame_stack=frame_stack, obs_type=obs_type,
        with_index=with_index,
    )
    buffer.extend(storage)
    return buffer
