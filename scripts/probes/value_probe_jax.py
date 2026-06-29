#!/usr/bin/env python
"""scripts/probes/value_probe_jax.py — value-function aliasing for GCIQL / CRL.

Cross-method companion to the FB-family value rung: does each method's value
function (the scalar the policy follows) separate success-bound from fail-bound
in-hand grasps? Per-task AUC on in-hand training states (success-bound = play-data
reach-before-end), averaged over the five eval tasks.

  GCIQL: V(s, g) = value(s, g)            (goal = eval-task goal observation)
  CRL:   Q(s, pi(s,g), g) = phi(s,a).psi(g) / sqrt(d)   (on-policy bilinear value)

Writes analysis/value/repsep/value_probe_<method>.json. Run under .venv-jax-cpu.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CONTACT = {"grasp", "transport"}
N_TASKS = 5


def _auc(y, s):
    """ROC-AUC = P(score(pos) > score(neg)) via the Mann-Whitney statistic."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, float)
    n1, n2 = int(y.sum()), int((~y).sum())
    if n1 < 5 or n2 < 5:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    R = ranks[y].sum()
    return float((R - n1 * (n1 + 1) / 2) / (n1 * n2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=["gciql", "crl"])
    ap.add_argument("--run-dir", help="GCIQL seed dir")
    ap.add_argument("--checkpoint-dir", help="CRL seed dir")
    ap.add_argument("--step", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--states",
                    default="analysis/value/training_value_multiseed/p0/training_states.npz")
    args = ap.parse_args()

    st = np.load(args.states, allow_pickle=True)
    obs = np.asarray(st["obs"], np.float32)
    region = np.array([str(r) for r in st["region"]])
    outcome = np.asarray(st["outcome"], bool)
    idx = np.where(np.isin(region, list(CONTACT)))[0]
    sub = obs[idx]

    import jax
    import jax.numpy as jnp

    if args.method == "gciql":
        assert args.run_dir, "--run-dir required"
        import ogbench
        from scripts.value.repsep_extract_jax import _load_gciql
        agent, _config, env_name = _load_gciql(args.run_dir, args.step)
        goals = {}
        for ti in range(1, N_TASKS + 1):
            env = ogbench.make_env_and_datasets(env_name, env_only=True)
            _o, info = env.reset(options=dict(task_id=ti))
            goals[ti] = np.asarray(info["goal"], np.float32)
            env.close()

        @jax.jit
        def value(O, G):
            v = jnp.asarray(agent.network.select("value")(O, G))
            return v.mean(0) if v.ndim > 1 else v   # twin -> mean

        per_task = []
        for ti in range(1, N_TASKS + 1):
            G = np.broadcast_to(goals[ti][None], sub.shape).astype(np.float32)
            v = np.asarray(value(sub, G))
            per_task.append(_auc(outcome[idx, ti - 1], v))

    else:  # crl
        assert args.checkpoint_dir, "--checkpoint-dir required"
        from scripts.probes.phase_probe_crl import EVAL_TASKS, load_agent, make_env
        agent = load_agent(Path(args.checkpoint_dir).resolve(), args.step, args.seed)
        goals = {}
        for ti, task in enumerate(EVAL_TASKS, start=1):
            env = make_env(task)
            _o, info = env.reset(seed=args.seed)
            g = info.get("goal")
            if g is None:
                from evals.phase_probe import episode_goal
                g = episode_goal(env)
            goals[ti] = np.asarray(g, np.float32)
            if hasattr(env, "close"):
                env.close()

        @jax.jit
        def qval(O, G, A):
            v = jnp.asarray(agent.network.select("critic")(O, G, A))
            return v.mean(0) if v.ndim > 1 else v   # ensemble -> mean

        per_task = []
        rng = jax.random.PRNGKey(args.seed)
        for ti in range(1, N_TASKS + 1):
            G = np.broadcast_to(goals[ti][None], sub.shape).astype(np.float32)
            rng, key = jax.random.split(rng)
            A = np.asarray(agent.sample_actions(observations=sub, goals=G,
                                                seed=key, temperature=0.0))
            A = np.clip(A, -1.0, 1.0).astype(np.float32)
            v = np.asarray(qval(sub, G, A))
            per_task.append(_auc(outcome[idx, ti - 1], v))

    per_task = np.array(per_task, float)
    out = REPO / f"analysis/value/repsep/value_probe_{args.method}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"method": args.method,
                               "V_mean": float(np.nanmean(per_task)),
                               "V_per_task": per_task.tolist()}, indent=2))
    print(f"[value_probe:{args.method}] per-task value AUC {per_task.round(3)} "
          f"-> mean {np.nanmean(per_task):.3f}  ({out})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
