"""Relabel a dataset's reward with the frozen readout r_hat(s') = phi(s')^T w.

One checkpoint, one w, one pass over every transition: the reward channel the zero-shot
agent DEPLOYS, written to disk so a training run can consume it through
`dataset.reward_override_path` with phi held fixed by construction. The 2026-09-13 audit
found the in-loop variants (Arm D1b's `phi_readout_fixed`) did not hold the reward fixed
because phi kept training under the held w; this file has no such path -- r_hat is a
constant array.

`w` is inferred exactly as tools/eval_checkpoint.py infers it: same global seed point,
`Dataset.sample(1)` for the example batch, then `eval_relabel_size` rows with the reward
shifted by `eval_reward_shift`, through `agent.infer_eval_z` (closed form
`project(E[r phi])` unless the run's config says `whitened`). No rescale is applied:
r_hat is phi^T w as the agent reads it.

Writes `<out>` (npz with `rewards`, float32, N) and `<out>.meta.json` (checkpoint, epoch,
w-inference settings, w itself, and the fit of r_hat against the real shifted reward:
Pearson correlation, R^2 raw and under the best affine rescale, top-1% precision).

Run:
  .venv/bin/python tools/relabel_reward_rhat.py \
      --run_dir $PSM_DATA/exp/PSMFLows/affine_strict_cube/sd001_* --epoch 500000 \
      --flow $PSM_DATA/flow/cube-single-play --flow_epoch 500000 \
      --env cube-single-play-singletask-v0 --out $PSM_DATA/rewards/<name>.npz
  `--limit N` relabels only the first N rows (a plumbing check; main.py refuses the file
  because the length no longer matches).
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax


def infer_w(agent, next_observations, rewards, reward_shift):
    """The eval task vector: `agent.infer_eval_z` on the shifted reward, as a numpy array."""
    z = agent.infer_eval_z(next_observations, rewards + float(reward_shift)).task_z
    return np.asarray(z, np.float32)


def rhat_in_batches(agent, next_observations, w, batch_size=20000):
    """r_hat = phi(s')^T w over every row, in slices so 1M rows never hit the device at once."""
    import jax.numpy as jnp
    w = jnp.asarray(w, jnp.float32)
    out = np.empty((len(next_observations),), np.float32)
    for lo in range(0, len(next_observations), batch_size):
        ph = agent.phi(jnp.asarray(next_observations[lo:lo + batch_size], jnp.float32))
        out[lo:lo + batch_size] = np.asarray(ph @ w, np.float32)
    return out


def relabel_stats(r_hat, r_true, reward_shift, top_frac=0.01):
    """How much of the real (shifted) reward the constant readout carries.

    `top_precision`: of the rows r_hat ranks in its top `top_frac`, the fraction whose real
    reward is above the minimum (a success row on cube's -1/0 reward); `base_rate` is the
    same fraction over all rows, i.e. what a random ranking would score.
    """
    r_hat = np.asarray(r_hat, np.float64).ravel()
    r = np.asarray(r_true, np.float64).ravel() + float(reward_shift)
    dh, dr = r_hat - r_hat.mean(), r - r.mean()
    corr = float((dh * dr).mean() / (dh.std() * dr.std() + 1e-12))
    ss_tot = float((dr ** 2).sum()) + 1e-12
    r2_raw = 1.0 - float(((r - r_hat) ** 2).sum()) / ss_tot
    a = float((dh * dr).sum() / ((dh ** 2).sum() + 1e-12))       # best affine rescale
    b = float(r.mean() - a * r_hat.mean())
    r2_affine = 1.0 - float(((r - (a * r_hat + b)) ** 2).sum()) / ss_tot
    hot = r > r.min()
    k = max(1, round(top_frac * len(r_hat)))
    top = np.argsort(-r_hat)[:k]
    return {"n": len(r_hat), "corr": corr, "r2_raw": r2_raw, "r2_affine": r2_affine,
            "affine_scale": a, "affine_offset": b,
            "rhat_mean": float(r_hat.mean()), "rhat_std": float(r_hat.std()),
            "rhat_min": float(r_hat.min()), "rhat_max": float(r_hat.max()),
            "reward_shifted_mean": float(r.mean()), "reward_shifted_std": float(r.std()),
            "top_frac": float(top_frac), "top_k": int(k),
            "top_precision": float(hot[top].mean()), "base_rate": float(hot.mean())}


def scale_to_real(r_hat, r_true, reward_shift):
    """`--match_real_scale`: the least-squares affine fit of r_hat to the SHIFTED reward,
    shifted back, so the output follows the dataset's own convention (-1/0 on cube: typical
    rows near -1, predicted successes near 0). Returns (r_out, scale, offset)."""
    s = relabel_stats(r_hat, r_true, reward_shift)
    r_out = s["affine_scale"] * np.asarray(r_hat, np.float64) + s["affine_offset"] - float(reward_shift)
    return r_out.astype(np.float32), s["affine_scale"], s["affine_offset"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="psmflow run dir (glob ok, one match)")
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--flow", required=True)
    ap.add_argument("--flow_epoch", type=int, default=500000)
    ap.add_argument("--env", default="cube-single-play-singletask-v0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0, help="eval seed (cfg.seed at eval)")
    ap.add_argument("--relabel_size", type=int, default=10000)
    ap.add_argument("--reward_shift", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="relabel only the first N rows")
    ap.add_argument("--match_real_scale", action="store_true",
                    help="write scale * r_hat + offset - reward_shift (least-squares fit to "
                         "the shifted reward, shifted back) instead of raw r_hat")
    a = ap.parse_args()

    import ml_collections

    from agents import agents
    from agents.psmflow import get_config
    from envs.env_utils import make_env_and_datasets
    from main import _lists_to_tuples
    from tools.eval_checkpoint import _cli_agent_keys, merge_run_config
    from utils.datasets import Dataset
    from utils.flax_utils import restore_agent

    runs = glob.glob(a.run_dir)
    assert len(runs) == 1, f"--run_dir matched {len(runs)}: {runs}"
    run_dir = runs[0].rstrip("/")

    # Same order as tools/eval_checkpoint.py::_evaluate_shard: global seed, dataset, one
    # example row, agent, restore, then the relabel batch -- so w is the eval's w.
    np.random.seed(a.seed)
    _, _, train_dataset, _ = make_env_and_datasets(a.env, frame_stack=None)
    ds = Dataset.create(**train_dataset)
    ex = ds.sample(1)

    base = json.loads(json.dumps(get_config().to_dict()))
    base["flow_ckpt_path"], base["flow_ckpt_epoch"] = a.flow, a.flow_epoch
    merged, prov = merge_run_config({"agent_name": "psmflow", **base}, run_dir, _cli_agent_keys())
    cfg = ml_collections.ConfigDict(_lists_to_tuples(merged))
    agent = agents["psmflow"].create(a.seed, ex["observations"], ex["actions"], cfg)
    agent = restore_agent(agent, run_dir, a.epoch)

    n_relabel = min(ds.size, int(a.relabel_size))
    zb = ds.sample(n_relabel)
    w = infer_w(agent, zb["next_observations"], zb["rewards"], a.reward_shift)
    fit_stats = relabel_stats(rhat_in_batches(agent, zb["next_observations"], w),
                              zb["rewards"], a.reward_shift)

    n = ds.size if a.limit <= 0 else min(ds.size, int(a.limit))
    next_obs = np.asarray(ds["next_observations"][:n], np.float32)
    r_true = np.asarray(ds["rewards"][:n], np.float32)
    r_hat = rhat_in_batches(agent, next_obs, w)
    assert np.isfinite(r_hat).all(), "non-finite r_hat"
    stats = relabel_stats(r_hat, r_true, a.reward_shift)

    r_out, output = r_hat, {"kind": "raw", "reward": "r_hat = phi(next_observations)^T w"}
    if a.match_real_scale:
        r_out, scale, offset = scale_to_real(r_hat, r_true, a.reward_shift)
        d_o, d_t = r_out - r_out.mean(), r_true - r_true.mean()
        output = {"kind": "scaled", "reward": "scale * r_hat + offset - reward_shift",
                  "affine_scale": scale, "affine_offset": offset,
                  "shift_back": float(a.reward_shift),
                  "mean": float(r_out.mean()), "std": float(r_out.std()),
                  "min": float(r_out.min()), "max": float(r_out.max()),
                  "corr_to_real": float((d_o * d_t).mean() / (d_o.std() * d_t.std() + 1e-12))}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, rewards=r_out)
    meta = {"kind": "reward_override", "output": output,
            "rhat": "r_hat = phi(next_observations)^T w",
            "env_name": a.env, "num_rows": int(n), "limit": int(a.limit),
            "dataset_rows": int(ds.size),
            "checkpoint": {"run_dir": run_dir, "epoch": int(a.epoch),
                           "flags_json": prov["flags_json"],
                           "flow_ckpt_path": a.flow, "flow_ckpt_epoch": int(a.flow_epoch)},
            "w_inference": {"seed": int(a.seed), "relabel_size": int(n_relabel),
                            "reward_shift": float(a.reward_shift),
                            "reward_inference": str(cfg["reward_inference"]),
                            "norm_z": bool(cfg["norm_z"]), "z_dim": int(cfg["z_dim"]),
                            "w_norm": float(np.linalg.norm(w)), "w": [float(x) for x in w]},
            "fit_on_relabel_batch": fit_stats, "fit_on_all_rows": stats}
    with open(a.out + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {a.out} ({n} rows, {output['kind']}) and .meta.json")
    if a.match_real_scale:
        print(f"scaled output: scale {output['affine_scale']:.6f} offset {output['affine_offset']:.6f}  "
              f"mean/std {output['mean']:.4f}/{output['std']:.4f}  min/max {output['min']:.4f}/"
              f"{output['max']:.4f}  corr to real {output['corr_to_real']:.4f}")
    print(f"relabel batch ({n_relabel} rows): corr {fit_stats['corr']:.4f}  "
          f"R2 raw {fit_stats['r2_raw']:.4f}  R2 affine {fit_stats['r2_affine']:.4f}")
    print(f"all {n} rows: corr {stats['corr']:.4f}  R2 raw {stats['r2_raw']:.4f}  "
          f"R2 affine {stats['r2_affine']:.4f}  top-{stats['top_frac']:.0%} precision "
          f"{stats['top_precision']:.4f} (base rate {stats['base_rate']:.4f})  "
          f"r_hat mean/std {stats['rhat_mean']:.4f}/{stats['rhat_std']:.4f} vs "
          f"reward {stats['reward_shifted_mean']:.4f}/{stats['reward_shifted_std']:.4f}")


if __name__ == "__main__":
    main()
