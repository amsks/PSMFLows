"""
scripts/eval/visualize_play.py — Replay offline cube-single episodes as a GIF.

Replays 4 episodes side-by-side (2x2 grid) by setting qpos from the
stored physics data at each timestep, then saves an animated GIF.

Usage:
    python scripts/eval/visualize_play.py
    python scripts/eval/visualize_play.py --episodes 0 42 99 200 --out outputs/play.gif
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ogbench.utils import make_env_and_datasets


def replay_episode(env, physics: np.ndarray, stride: int) -> list[np.ndarray]:
    """Set qpos from physics[::stride] and render each frame."""
    frames = []
    for t in range(0, len(physics), stride):
        env.unwrapped.data.qpos[:] = physics[t]
        env.unwrapped.data.qvel[:] = 0.0
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
        frames.append(env.render().copy())
    return frames


def label_frame(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.text((6, 6), text, fill=(255, 255, 255))
    return np.array(img)


def tile_2x2(f0, f1, f2, f3) -> np.ndarray:
    top = np.concatenate([f0, f1], axis=1)
    bot = np.concatenate([f2, f3], axis=1)
    return np.concatenate([top, bot], axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_path", default="datasets")
    ap.add_argument("--domain", default="cube-single-play-v0")
    ap.add_argument("--episodes", type=int, nargs=4, default=[0, 42, 99, 200],
                    metavar="EP", help="4 episode indices to visualize")
    ap.add_argument("--stride", type=int, default=5,
                    help="Render every N-th timestep (default: 5 → 200 frames/episode)")
    ap.add_argument("--render_size", type=int, default=256)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default="outputs/play_trajectories.gif")
    args = ap.parse_args()

    buf_dir = Path(args.data_path) / args.domain / "buffer"
    all_files = sorted(buf_dir.glob("*.npz"))
    if not all_files:
        raise FileNotFoundError(f"No episodes found at {buf_dir}")

    print(f"Loading env for rendering ({args.render_size}x{args.render_size})...")
    env = make_env_and_datasets(
        args.domain, env_only=True,
        height=args.render_size, width=args.render_size,
    )
    env.reset()

    # Load the 4 chosen episodes
    episodes = []
    for idx in args.episodes:
        f = all_files[idx]
        ep = np.load(f)
        physics = ep["physics"].astype(np.float32)
        episodes.append((idx, physics))
        print(f"  Episode {idx:4d}: {f.name}  T={len(physics)}")

    # Replay each episode
    print(f"\nReplaying (stride={args.stride})...")
    episode_frames = []
    for ep_idx, (idx, physics) in enumerate(episodes):
        print(f"  Episode {idx}...")
        frames = replay_episode(env, physics, stride=args.stride)
        labeled = [label_frame(f, f"ep {idx:03d}  t={t*args.stride:4d}")
                   for t, f in enumerate(frames)]
        episode_frames.append(labeled)
        print(f"    {len(labeled)} frames rendered")

    env.close()

    # Tile into 2x2 grid and save GIF
    n_frames = min(len(e) for e in episode_frames)
    print(f"\nCompositing {n_frames} tiled frames...")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pil_frames = []
    for i in range(n_frames):
        tiled = tile_2x2(
            episode_frames[0][i],
            episode_frames[1][i],
            episode_frames[2][i],
            episode_frames[3][i],
        )
        pil_frames.append(Image.fromarray(tiled).quantize(256))

    duration_ms = int(1000 / args.fps)
    pil_frames[0].save(
        args.out,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=duration_ms,
    )
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"\nSaved: {args.out}  ({size_mb:.1f} MB, {len(pil_frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
