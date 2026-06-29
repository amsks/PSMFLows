"""
scripts/dev/smoke.py — Synthetic-data pipeline smoke test (no OGBench data needed).

Builds FBAgent and FBFlowBCAgent from the Hydra configs and runs a few update
steps on random Box observations / actions. Verifies the math doesn't blow up
and the config composition is correct.

Usage:
    python scripts/dev/smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run from anywhere
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from gymnasium.spaces import Box
from hydra import compose, initialize_config_dir

from train import make_agent


def smoke(domain: str, expected_type: str, *, batch_size: int = 32, n_steps: int = 5) -> None:
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[f"domain={domain}", "device=cpu", f"batch_size={batch_size}"],
        )

    obs_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
    agent = make_agent(cfg, obs_space, action_dim=4)
    assert type(agent).__name__ == expected_type, (type(agent).__name__, expected_type)

    last = None
    for step in range(n_steps):
        batch = {
            "observation": torch.randn(batch_size, 15),
            "action": torch.randn(batch_size, 4).clamp(-1, 1),
            "next": {
                "observation": torch.randn(batch_size, 15),
                "terminated": torch.zeros(batch_size, 1),
            },
        }
        last = agent.update(batch, step)

    assert all(np.isfinite(v) for v in last.values()), last
    print(
        f"OK {domain:18s} {expected_type:14s} "
        f"fb_loss={last['fb_loss']:.2f} "
        f"actor_loss={last['actor_loss']:.2f}"
    )


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    # td_jepa uses FBFlowBCAgent on ALL OGBench domains (bc_coeff=0.3 antmaze, 3.0 cube).
    smoke("antmaze_medium", "FBFlowBCAgent")
    smoke("antmaze_large",  "FBFlowBCAgent")
    smoke("cube_single",    "FBFlowBCAgent")
    smoke("scene",          "FBFlowBCAgent")
    smoke("puzzle_3x3",     "FBFlowBCAgent")
    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
