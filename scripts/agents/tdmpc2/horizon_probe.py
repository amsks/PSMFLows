"""horizon_probe.py — diagnostic: does longer MPC planning horizon recover any
eval success from an already-trained TDMPC2 checkpoint?

The planning horizon (MPPI rollout length in core._plan / _estimate_value) is a
RUNTIME knob, not a learned weight — so we can load a frozen checkpoint and
re-evaluate at several horizons without retraining. The world model was only
*trained* for accurate 3-step rollouts, so longer horizons plan on increasingly
extrapolated dynamics. The result discriminates the failure mode:

  * success rises with horizon  -> planner was myopic; world model is usable;
                                   retraining at a longer training-horizon is worth it.
  * success stays at 0 / -200   -> bottleneck is world-model / Q quality, not
                                   horizon; longer horizon won't help.

Usage:
    python scripts/agents/tdmpc2/horizon_probe.py \
        --ckpt /dev/shm/factored-fb/runs/<run>/checkpoints/step_400000.pt \
        --domain cube-single-play-v0 --horizons 3,8,16 --episodes 5 --device cuda
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import torch

from agents.tdmpc2 import TDMPC2Agent
from envs.ogbench import ALL_TASKS, create_ogbench_env
from evals.ogbench import OGBenchEvaluator


def set_horizon(agent: TDMPC2Agent, H: int) -> None:
    """Repoint the planner at a new horizon on an already-built agent.

    Only the MPPI loop and the _prev_mean warm-start buffer depend on horizon;
    the dynamics/reward/Q nets operate per-step and are dimension-agnostic to it.
    """
    agent.cfg.horizon = int(H)          # same OmegaConf object the core holds
    agent.horizon = int(H)
    agent.core._plan_val = None          # drop cached bound/compiled plan fn
    agent.core._prev_mean = torch.nn.Buffer(
        torch.zeros(int(H), agent.cfg.action_dim, device=agent.device)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--domain", default="cube-single-play-v0")
    ap.add_argument("--horizons", default="3,8,16")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    # Build the env once just to read obs/action dims (matches train.py).
    env, _ = create_ogbench_env(args.domain, obs_type="state", seed=args.seed)
    obs_space = env.observation_space
    action_dim = env.action_space.shape[0]
    env.close()
    print(f"[probe] obs_space={obs_space.shape} action_dim={action_dim} device={device}")

    # Build agent at the largest horizon so _prev_mean starts big enough; we
    # reset it per-horizon anyway. batch_size is irrelevant for eval-only.
    agent = TDMPC2Agent(
        obs_space, action_dim, device=device,
        horizon=max(horizons), batch_size=256,
    )
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    agent.load_state_dict(state)
    print(f"[probe] loaded checkpoint: {args.ckpt}")

    tasks = ALL_TASKS[args.domain]
    print(f"[probe] {len(tasks)} tasks, {args.episodes} episodes each\n")

    for H in horizons:
        set_horizon(agent, H)
        # offline_buffer=None is safe: TDMPC2 uses eval_context (env goal), not _infer_z.
        evaluator = OGBenchEvaluator(
            domain=args.domain, agent=agent, offline_buffer=None,
            n_episodes=args.episodes, obs_type="state", seed=args.seed,
            device=device, save_videos=False,
        )
        metrics = evaluator.run(step=0)
        succ = metrics.get("eval/success", float("nan"))
        rew = metrics.get("eval/reward", float("nan"))
        per_task = "  ".join(
            f"t{i+1}:{metrics.get(f'{t}/success', 0.0):.2f}"
            for i, t in enumerate(tasks)
        )
        print(f"[probe] horizon={H:>2d}  eval/success={succ:.3f}  eval/reward={rew:8.2f}  | {per_task}")


if __name__ == "__main__":
    main()
