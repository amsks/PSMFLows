"""Does the agent that fails actually SUCCEED at the reward it was trained on?

Every arm that optimises the fitted reward well ends below the behaviour-cloning control:
D1b 0.011, Arm C + gradient actor 0.000, D2 0.004, the 09-08 faithful arm 0.14. Best-of-64
GPI survives at 0.38-0.50. The reading on offer is that these are not optimisation failures
at all -- the optimiser is succeeding, at a reward that is not the task. That is an inference
from a pattern of numbers. This measures it.

Per episode, under the SAME rollout, record two discounted returns:

  true      sum_t gamma^t r_t                       -- the task's own reward
  fitted    sum_t gamma^t rhat(s_{t+1})             -- rhat = scale * phi(s')^T w

`w` is the deployed task vector: the closed form on an `eval_relabel_size` relabel batch of
SHIFTED rewards, exactly as `tools/eval_checkpoint.py:350` builds it. `scale` matches rhat's
std to the shifted reward's over that batch, as `dsrl_na.reward_source=phi_readout_fixed`
does, so the two returns are on one scale and their ratio means something.

Read: an agent whose fitted return is at or above a working agent's while its true success is
~0 is optimising the fitted reward successfully and the fitted reward is wrong. An agent low
on both is simply not optimising.
"""
import argparse
import json
import os

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="comma-separated <label>=<run dir>@<epoch> triples")
    ap.add_argument("--flow", required=True)
    ap.add_argument("--flow_epoch", type=int, default=500000)
    ap.add_argument("--preimages", default="")
    ap.add_argument("--env", default="cube-single-play-singletask-v0")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--relabel", type=int, default=10000)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report_out", required=True)
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp

    from agents import agents
    from agents.psmflow import get_config
    from envs.env_utils import make_env_and_datasets
    from tools.eval_checkpoint import merge_run_config
    from utils.datasets import Dataset
    from utils.flax_utils import restore_agent

    env, _, train_dataset, _ = make_env_and_datasets(a.env, frame_stack=None, add_info=True)
    ds = Dataset.create(**train_dataset)
    ex = ds.sample(1)
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(ds.size, min(a.relabel, ds.size), replace=False))
    rl_next = jnp.asarray(np.asarray(ds["next_observations"][idx], np.float32))
    rl_rew = jnp.asarray(np.asarray(ds["rewards"][idx], np.float32).ravel()) + a.shift

    out = {"probe": "true vs fitted discounted return under the same rollout",
           "env": a.env, "episodes": a.episodes, "reward_shift": a.shift,
           "relabel_rows": len(idx), "runs": {}}

    for spec in a.runs.split(","):
        label, rest = spec.split("=", 1)
        run_dir, epoch = rest.rsplit("@", 1)
        epoch = int(epoch)

        base = json.loads(json.dumps(get_config().to_dict()))
        cfg, _ = merge_run_config(base, run_dir, {})
        cfg["flow_ckpt_path"], cfg["flow_ckpt_epoch"] = a.flow, a.flow_epoch
        cfg["preimage_path"] = a.preimages or None
        agent = agents["psmflow"].create(a.seed, ex["observations"], ex["actions"], cfg)
        agent = restore_agent(agent, run_dir, epoch)
        agent = agent.infer_eval_z(rl_next, rl_rew)          # the DEPLOYED w

        # rhat's scale, matched to the shifted reward over the same relabel batch. Without
        # this the two returns differ by ||w||'s arbitrary sphere radius and no ratio is
        # meaningful.
        w = agent.task_z
        rhat_rl = agent.phi(rl_next) @ w
        scale = float(rl_rew.std() / (rhat_rl.std() + 1e-8))
        gamma = float(cfg["discount"])

        true_r, fit_r, succ = [], [], []
        for ep in range(a.episodes):
            ob, _ = env.reset(seed=a.seed * 10000 + ep)
            done, t, tr, fr, ok = False, 0, 0.0, 0.0, 0.0
            while not done:
                act = agent.sample_actions(ob, seed=jax.random.PRNGKey(a.seed * 99991 + t))
                act = np.clip(np.asarray(act), -1, 1)
                ob, r, term, trunc, info = env.step(act)
                rh = float(agent.phi(jnp.asarray(ob)[None])[0] @ w) * scale
                tr += (gamma ** t) * float(r)
                fr += (gamma ** t) * rh
                ok = max(ok, float(info.get("success", 0.0)))
                done, t = bool(term or trunc), t + 1
            true_r.append(tr); fit_r.append(fr); succ.append(ok)

        rec = {"run_dir": run_dir, "epoch": epoch, "discount": gamma, "rhat_scale": scale,
               "true_return_mean": float(np.mean(true_r)),
               "true_return_sd": float(np.std(true_r)),
               "fitted_return_mean": float(np.mean(fit_r)),
               "fitted_return_sd": float(np.std(fit_r)),
               "success_mean": float(np.mean(succ))}
        out["runs"][label] = rec
        print(f"{label:22s} success={rec['success_mean']:.3f}  "
              f"true={rec['true_return_mean']:+8.3f}  "
              f"fitted={rec['fitted_return_mean']:+8.3f}", flush=True)

    ref = out["runs"].get("dsrlna_real")
    if ref:
        for v in out["runs"].values():
            v["fitted_over_dsrlna_real"] = (
                v["fitted_return_mean"] / ref["fitted_return_mean"]
                if ref["fitted_return_mean"] else None)

    os.makedirs(os.path.dirname(os.path.abspath(a.report_out)), exist_ok=True)
    with open(a.report_out, "w") as f:
        json.dump(out, f, indent=2)
    print("report ->", a.report_out)


if __name__ == "__main__":
    main()
