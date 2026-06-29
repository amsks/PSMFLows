"""latent_probe.py — diagnostic: did the TDMPC2 world model collapse, or does
its latent actually encode (and propagate) cube position?

Decoder-free TD-MPC2 cannot reconstruct observations, so we probe the latent:
fit a closed-form ridge probe z0 -> cube_xyz on encoded offline states, then
roll the *learned* dynamics (model.next) open-loop and watch the probe's
cube-xyz error grow. Reads:

  * cube_xyz_mse (single-step)   -- can the latent linearly decode cube xyz at all?
                                    huge MSE => latent collapse / cube not represented.
  * open_loop_cube_mse_by_step   -- does model.next preserve/predict cube motion,
                                    or does error explode after 1-2 imagined steps?

Pairs with horizon_probe.py: horizon tests the planner, this tests the model.

Usage:
    python scripts/agents/tdmpc2/latent_probe.py \
        --ckpt /dev/shm/factored-fb/runs/<run>/checkpoints/step_1000000.pt \
        --domain cube-single-play-v0 --batch 4096 --rollout-len 3 --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from agents.tdmpc2 import TDMPC2Agent
from data.ogbench import load_ogbench_dataset
from envs.ogbench import create_ogbench_env
from evals.phase_probe import tdmpc2_latent_phase_probe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--domain", default="cube-single-play-v0")
    ap.add_argument("--data-path", default="/dev/shm/factored-fb/datasets")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--rollout-len", type=int, default=3)
    ap.add_argument("--load-n-episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    env, _ = create_ogbench_env(args.domain, obs_type="state", seed=args.seed)
    obs_space = env.observation_space
    action_dim = env.action_space.shape[0]
    env.close()

    agent = TDMPC2Agent(
        obs_space, action_dim, device=device,
        horizon=max(args.rollout_len, 3), batch_size=args.batch,
    )
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    agent.load_state_dict(state)
    print(f"[latent] loaded {args.ckpt}")

    buffer = load_ogbench_dataset(
        domain=args.domain, data_path=args.data_path,
        load_n_episodes=args.load_n_episodes, device=device,
        n_transitions=None, obs_type="state", frame_stack=1,
    )
    batch = buffer.sample(args.batch, horizon=max(args.rollout_len, 3))

    # Goal folded into the encoder: use the window-start achieved cube xyz so the
    # probe sees the same (obs, goal) construction the agent trains/acts on.
    goal = batch["next"]["physics"][:, 0, 14:17].mean(dim=0)  # [3] representative goal

    # Diagnostic on the latent's z->cube decodability and z-collapse.
    z0_std = None
    with torch.no_grad():
        folded = agent._fold(batch["observation"][:, 0, :].to(device).float(),
                             goal.to(device).float().expand(args.batch, 3))
        z0 = agent.core.model.encode(folded, None)
        z0_std = z0.std(dim=0).mean().item()      # ~0 => latent collapsed across samples
        cube_var = batch["next"]["physics"][:, 0, 14:17].var(dim=0).mean().item()

    out = tdmpc2_latent_phase_probe(agent, batch, goal, rollout_len=args.rollout_len)

    print(f"[latent] batch={args.batch}  latent_std(mean over dims)={z0_std:.5f}  "
          f"(near 0 => collapse)")
    print(f"[latent] cube_xyz target variance (data) = {cube_var:.5f}")
    print(f"[latent] single-step probe cube_xyz_mse = {out['cube_xyz_mse']:.6f}")
    print(f"[latent] open-loop cube_xyz_mse by imagined step:")
    for t, m in enumerate(out["open_loop_cube_mse_by_step"]):
        print(f"           step {t}: {m:.6f}")


if __name__ == "__main__":
    main()
