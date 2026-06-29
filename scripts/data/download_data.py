"""
scripts/data/download_data.py — Download OGBench datasets and save locally.

OGBench provides datasets via make_env_and_datasets() which auto-downloads
on first call.  This script triggers that download and re-saves the data in
the episode .npz format that data/ogbench.py expects:

    <data_path>/<domain>/buffer/ep0000.npz
    <data_path>/<domain>/buffer/ep0001.npz
    ...

Each .npz stores one episode with keys:
    observation  [T, obs_dim]
    action       [T, act_dim]
    physics      [T, phys_dim]  (full qpos — needed for reward relabelling)
    discount     [T]            (0.0 at episode-terminal steps)

Usage
-----
# Download all OGBench domains to ./datasets/
python scripts/data/download_data.py

# Specific domains only
python scripts/data/download_data.py --domains antmaze-medium-navigate-v0 antmaze-large-navigate-v0

# Custom output path
python scripts/data/download_data.py --data_path /scratch/ogbench_data

# Only download, skip conversion to episode format
python scripts/data/download_data.py --raw_only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from envs.ogbench import ALL_DOMAINS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--domains", nargs="+", default=None,
        help="Domains to download.  Default: all OGBench domains.",
    )
    p.add_argument(
        "--data_path", type=str, default="datasets/",
        help="Root output directory.",
    )
    p.add_argument(
        "--raw_only", action="store_true",
        help="Just trigger the OGBench download; do not convert to episode format.",
    )
    p.add_argument(
        "--episodes_per_domain", type=int, default=1000,
        help="Maximum number of synthetic episodes to create per domain.",
    )
    return p.parse_args()


def download_and_convert(domain: str, data_path: str, max_episodes: int) -> None:
    """Download one domain via OGBench and save as episode .npz files."""
    from ogbench.utils import make_env_and_datasets

    print(f"[download] {domain} …")
    # make_env_and_datasets auto-downloads on first call
    env, train_ds, _ = make_env_and_datasets(domain)

    obs        = train_ds["observations"]         # [N, obs_dim]
    next_obs   = train_ds["next_observations"]    # [N, obs_dim]
    actions    = train_ds["actions"]              # [N, act_dim]
    terminals  = train_ds["terminals"]            # [N]  bool/int
    # OGBench may expose physics/qpos under different keys depending on version
    physics    = _get_physics(train_ds, env, obs)  # [N, phys_dim] or None

    # Split flat transitions back into episodes at terminal boundaries
    episodes = _split_episodes(obs, next_obs, actions, terminals, physics)
    print(f"  → {len(episodes)} episodes,  "
          f"{sum(len(e['observation']) for e in episodes):,} transitions")

    # Save
    out_dir = Path(data_path) / domain / "buffer"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_save = min(len(episodes), max_episodes)
    for i, ep in enumerate(tqdm(episodes[:n_save], desc=f"  saving {domain}", leave=False)):
        np.savez(out_dir / f"ep{i:04d}.npz", **ep)

    print(f"  ✓ saved {n_save} episodes to {out_dir}")


def _get_physics(dataset: dict, env, obs: np.ndarray) -> np.ndarray | None:
    """Extract physics/qpos from dataset.  Key name varies across OGBench versions."""
    for key in ("physics", "qpos", "proprio", "infos/qpos"):
        if key in dataset:
            return dataset[key].astype(np.float32)
    # Fallback: use obs as physics (works for antmaze where obs ≈ qpos)
    print("  [warn] no 'physics' key found — using obs as physics fallback")
    return obs.astype(np.float32)


def _split_episodes(
    obs: np.ndarray,
    next_obs: np.ndarray,
    actions: np.ndarray,
    terminals: np.ndarray,
    physics: np.ndarray | None,
) -> list[dict]:
    """Convert flat transitions to a list of episode dicts.

    Each episode is reconstructed as T+1 observations from T transitions:
        observation[0..T-1] = obs[start..end]
        observation[T]      = next_obs[end]    (final observation)
        action[0..T-1]      = actions[start..end]
        physics[0..T-1]     = physics[start..end]
        discount[t]         = 0.0 if terminals[t] else 1.0
    """
    episodes = []
    start = 0
    N = len(obs)

    terminal_indices = np.where(terminals.astype(bool))[0]
    # Ensure the last transition is treated as a terminal
    if len(terminal_indices) == 0 or terminal_indices[-1] != N - 1:
        terminal_indices = np.append(terminal_indices, N - 1)

    for end in terminal_indices:
        ep_obs  = np.concatenate([obs[start:end + 1], next_obs[end:end + 1]], axis=0)
        ep_act  = actions[start:end + 1]
        ep_disc = np.ones(end - start + 2, dtype=np.float32)
        ep_disc[-1] = 0.0   # terminal

        ep: dict = {
            "observation": ep_obs.astype(np.float32),
            "action":      ep_act.astype(np.float32),
            "discount":    ep_disc,
        }
        if physics is not None:
            ep_phys = np.concatenate(
                [physics[start:end + 1], physics[end:end + 1]], axis=0
            )
            ep["physics"] = ep_phys.astype(np.float32)

        episodes.append(ep)
        start = end + 1
        if start >= N:
            break

    return episodes


def main() -> None:
    args = parse_args()
    domains = args.domains or ALL_DOMAINS

    print(f"Downloading {len(domains)} domain(s) → {args.data_path}")
    for domain in domains:
        try:
            if args.raw_only:
                from ogbench.utils import make_env_and_datasets
                print(f"[download] {domain} (raw only) …")
                make_env_and_datasets(domain)
                print(f"  ✓ cached")
            else:
                download_and_convert(domain, args.data_path, args.episodes_per_domain)
        except Exception as e:
            print(f"  ✗ {domain}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()