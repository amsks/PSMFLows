#!/usr/bin/env python3
"""Offline re-eval of saved PSM checkpoints with the SEEDED eval (post eval-env-seeding fix).

Loads each seed's params_<step>.pkl, rebuilds the agent from its flags.json (so the
param tree matches), infers the eval-z exactly as main.py does, then runs the NEW
reproducible eval (evaluate(..., seed=cfg.seed)) and prints task2 success next to the
code-matched reference value at the same step/seed.

Usage:
  CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
    PYTHONPATH=. .venv/bin/python scripts/reeval_checkpoints.py \
      --group psm_recover500k_flow_ortho1000_20260713_160709 --step 500000 --seeds 0 1 2 \
      --episodes 10 50
"""
import argparse
import glob
import json
import os
import random

import ml_collections
import numpy as np

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent

BASE = "/var/local/amsks/exp/PSMFLows"
REF = json.load(open("/var/local/amsks/exp/ref_orthohi_task2_curves.json"))


def _success(stats):
    """evaluate() returns flattened info keys (main.py adds the 'evaluation/' prefix
    later). Single-task cube => 'success' is task2 success."""
    for k in ("success", "evaluation/success"):
        if k in stats:
            return float(stats[k])
    cand = [k for k in stats if k.split("/")[-1] == "success"]
    if not cand:
        raise KeyError(f"no success key in eval stats; keys={list(stats)}")
    return float(stats[cand[0]])


def _lists_to_tuples(x):
    if isinstance(x, list):
        return tuple(_lists_to_tuples(v) for v in x)
    if isinstance(x, dict):
        return {k: _lists_to_tuples(v) for k, v in x.items()}
    return x


def ref_at(seed, step):
    ok = [v for st, v in REF.get(str(seed), []) if st <= step]
    return ok[-1] if ok else None


def reeval_one(group, seed, step, episodes):
    run_dir = sorted(glob.glob(f"{BASE}/{group}/sd{seed:03d}_*"))[-1]
    flags = json.load(open(f"{run_dir}/flags.json"))
    config = ml_collections.ConfigDict(_lists_to_tuples(flags["agent"]))
    env_name = flags["env_name"]
    frame_stack = flags.get("frame_stack")
    relabel = int(flags.get("eval_relabel_size", 10000))
    shift = float(flags.get("eval_reward_shift", 1.0))

    # Reproducible: seed exactly as main.py does before dataset/eval work.
    random.seed(seed)
    np.random.seed(seed)

    env, eval_env, train_dataset, _ = make_env_and_datasets(env_name, frame_stack=frame_stack)

    ex = train_dataset.sample(1)
    agent = agents[config["agent_name"]].create(seed, ex["observations"], ex["actions"], config)
    agent = restore_agent(agent, run_dir, step)

    # Eval-z inference — identical to main.py:197-200.
    n = min(train_dataset.size, relabel)
    zb = train_dataset.sample(n)
    rew = zb["rewards"] + shift
    eval_agent = agent.infer_eval_z(zb["next_observations"], rew)

    out = {}
    for nep in episodes:
        stats, _, _ = evaluate(eval_agent, eval_env, config=config,
                               num_eval_episodes=nep, seed=seed)
        out[nep] = _success(stats)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="psm_recover500k_flow_ortho1000_20260713_160709")
    ap.add_argument("--step", type=int, default=500000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, nargs="+", default=[10, 50])
    args = ap.parse_args()

    rows = []
    for seed in args.seeds:
        res = reeval_one(args.group, seed, args.step, args.episodes)
        rows.append((seed, res, ref_at(seed, args.step)))

    print("\n" + "=" * 68)
    print(f"SEEDED re-eval @ {args.step//1000}k  (group {args.group})")
    print("=" * 68)
    hdr = "seed | " + " ".join(f"ours@{n}ep" for n in args.episodes) + " | ref@" + str(args.step // 1000) + "k | diff(10ep)"
    print(hdr)
    print("-" * len(hdr))
    for seed, res, ref in rows:
        ours = " ".join(f"{res[n]:>8.2f}" for n in args.episodes)
        rref = f"{ref:.2f}" if ref is not None else "n/a"
        d = res[args.episodes[0]] - ref if ref is not None else float("nan")
        print(f"{seed:>4} | {ours} | {rref:>7} | {d:+.2f}")


if __name__ == "__main__":
    main()
