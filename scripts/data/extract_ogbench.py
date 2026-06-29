"""
scripts/data/extract_ogbench.py — Split OGBench's monolithic .npz datasets into
per-episode .npz files under <output>/<out_env>/buffer/episode_XXXXXX_<len>.npz.

Faithful adaptation of td_jepa's scripts/data_processing/ogbench/extract_all.py.
State envs write a real "observation" (no "pixels" key). Visual envs write a
dummy zero "observation" and a "pixels" array. DEVIATION FROM td_jepa: td_jepa
stores pixels CHW (np.moveaxis(...,-1,1)); we store **HWC** so this repo's
data/ogbench.py::load_transitions (which does its own HWC->CHW moveaxis) is
correct with the loader left untouched.

Usage:
    python scripts/data/extract_ogbench.py --output_folder datasets
    python scripts/data/extract_ogbench.py --env cube-single-play-v0 --output_folder datasets
    python scripts/data/extract_ogbench.py --env visual-cube-single-play-v0 \\
        --out_env cube-single-play-v0 \\
        --output_folder /dev/shm/factored-fb/datasets \\
        --cache_dir /dev/shm/factored-fb/ogbench_cache

--out_env overrides the output env-folder name (default: the env name; matches
td_jepa). --cache_dir redirects the OGBench monolithic download off the default
~/.ogbench cache (needed when the root disk is full). By default downloads via
ogbench.utils.make_env_and_datasets then splits into per-episode files.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Honor an existing MUJOCO_GL; otherwise pick a sane per-platform default
# (macOS has no EGL — forcing egl there breaks env creation during download).
os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

import numpy as np
from tqdm import tqdm


ALL_ENVS_STATE = [
    "antmaze-medium-navigate-v0",
    "antmaze-large-navigate-v0",
    "antmaze-giant-navigate-v0",
    "antmaze-medium-stitch-v0",
    "antmaze-large-stitch-v0",
    "antmaze-medium-explore-v0",
    "antmaze-large-explore-v0",
    "cube-single-play-v0",
    "cube-double-play-v0",
    "scene-play-v0",
    "puzzle-3x3-play-v0",
]


ALL_ENVS_VISUAL = [
    "visual-antmaze-medium-navigate-v0",
    "visual-antmaze-large-navigate-v0",
    "visual-antmaze-medium-stitch-v0",
    "visual-antmaze-large-stitch-v0",
    "visual-antmaze-medium-explore-v0",
    "visual-antmaze-large-explore-v0",
    "visual-cube-single-play-v0",
    "visual-cube-double-play-v0",
    "visual-scene-play-v0",
    "visual-puzzle-3x3-play-v0",
]


def extract(
    env_name: str,
    output_folder: str,
    dataset_path: str | None = None,
    cache_dir: str | None = None,
    out_env: str | None = None,
) -> None:
    from ogbench.utils import DEFAULT_DATASET_DIR, make_env_and_datasets

    if dataset_path is not None:
        dataset_file = Path(dataset_path) / (env_name + ".npz")
    else:
        # Triggers download into the chosen cache (default ~/.ogbench) if needed.
        if cache_dir is not None:
            make_env_and_datasets(env_name, dataset_dir=cache_dir)
            base = Path(cache_dir).expanduser()
        else:
            make_env_and_datasets(env_name)
            base = Path(DEFAULT_DATASET_DIR).expanduser()
        dataset_file = base / (env_name + ".npz")

    print(f"[{env_name}] dataset: {dataset_file}")
    out_env = out_env if out_env is not None else env_name
    out_dir = Path(output_folder) / out_env / "buffer"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{env_name}] writing episodes to: {out_dir}")

    data = np.load(dataset_file)
    data = {k: data[k] for k in data.keys()}

    nz = data["terminals"].ravel().nonzero()[0]
    ends = np.arange(data["terminals"].shape[0])[nz]
    starts = np.concatenate(([0], ends[:-1] + 1))
    lengths = ends - starts

    is_visual = "visual" in env_name
    for ep_idx, (start, length, end) in enumerate(tqdm(list(zip(starts, lengths, ends)))):
        ep = {
            "action": data["actions"][start : end + 1].copy(),
            "physics": data["qpos"][start : end + 1],
            "reward": np.zeros((length + 1, 1), dtype=np.float32),
            "discount": np.ones((length + 1, 1), dtype=np.float32),
        }
        if is_visual:
            # td_jepa-faithful: dummy state obs; pixels carry the images.
            # DEVIATION: store HWC (no np.moveaxis) so load_transitions'
            # HWC->CHW moveaxis is correct and the loader stays untouched.
            ep["observation"] = np.zeros((length + 1, 1), dtype=np.float32)
            ep["pixels"] = data["observations"][start : end + 1]
        else:
            ep["observation"] = data["observations"][start : end + 1]
        # Shift actions so action[t] is the action that led to observation[t].
        # First-step action is set to zero (matches td_jepa convention).
        ep["action"][1:] = ep["action"][:-1]
        ep["action"][0] = 0.0
        # NOTE: td_jepa's extract_all.py leaves discount all-ones and never
        # marks a terminal step; the loader treats every transition as
        # non-terminal (TD bootstrap always uses discount = gamma).
        # Some envs have extra button states concatenated into physics.
        if "button_states" in data:
            ep["physics"] = np.concatenate(
                [ep["physics"], data["button_states"][start : end + 1]], axis=-1
            )

        filename = f"episode_{ep_idx:06d}_{length}.npz"
        np.savez_compressed(out_dir / filename, **ep)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_folder", default="datasets",
                    help="Where to write <out_env>/buffer/episode_*.npz (default: datasets)")
    ap.add_argument("--dataset_path", default=None,
                    help="Optional: dir containing a pre-downloaded <env>.npz. "
                         "If omitted, ogbench downloads to --cache_dir (or its default).")
    ap.add_argument("--cache_dir", default=None,
                    help="Redirect the OGBench download cache (passed as "
                         "make_env_and_datasets(dataset_dir=...)). Use this when "
                         "the default ~/.ogbench location is on a full disk.")
    ap.add_argument("--out_env", default=None,
                    help="Override the output env-folder name (default: --env). "
                         "E.g. --env visual-cube-single-play-v0 "
                         "--out_env cube-single-play-v0 so the committed loader "
                         "path datasets/<domain>/buffer resolves.")
    ap.add_argument("--env", default=None,
                    help="A single OGBench env (state or visual). "
                         "If omitted, extracts all state envs.")
    args = ap.parse_args()

    envs = [args.env] if args.env else ALL_ENVS_STATE
    for env_name in envs:
        extract(env_name, args.output_folder, args.dataset_path,
                cache_dir=args.cache_dir, out_env=args.out_env)


if __name__ == "__main__":
    main()
