"""Relabel a dataset's reward from a psmgoal checkpoint's goal-set measure readout.

The psmflow sibling (tools/relabel_reward_rhat.py) reads r_hat = phi(s')^T w. psmgoal's
measure is a triple M(s, u, g), so the readout is different and this tool is separate; the
psmflow path is left byte-identical. Two readouts:

  goal_set (primary): r_hat(s', u') = mean over g in the eval goal set G of
      M(s', u', g) = phi(s',u',g)^T w* + b(s',u',g)
    with u' the row's flow preimage (the point preimage if use_point_preimage else the
    mixture mean), clipped to +-u_clip. G and w* come from psmgoal.infer_eval_goals on a
    relabel batch, exactly as tools/eval_checkpoint.py builds them.

  regression (control): r_hat(s') = f(s')^T w, w = lstsq(f, real_reward), f(s) the
    goal/action-marginalised psmgoal state feature (utils.psm_networks.PsmgoalProjectedPhi).
    This isolates feature quality from the learned coefficient w*(g).

Scale to the dataset -1/0 convention with --match_real_scale (the same least-squares fit as
the psmflow tool, reused), then write a reward_override npz + .meta.json a DSRL-NA scalar
critic can consume via dataset.reward_override_path.

The agent is rebuilt from the run's own flags.json (flow_ckpt_path/epoch, preimage_path,
net widths, u_clip, coef_source), so nothing about the architecture is guessed here.

Run:
  JAX_PLATFORMS=cpu OGBENCH_DATASET_DIR=... PSM_DATA=... \
  .venv/bin/python tools/relabel_reward_rhat_psmgoal.py \
      --run_dir $PSM_DATA/exp/PSMFLows/psmgoal_lift_cube_gc/sd000_* --epoch 750000 \
      --env cube-single-play-singletask-v0 --readout goal_set --match_real_scale \
      --out $PSM_DATA/rewards/<name>.npz
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax

# The scale-to-real fit and the fit statistics are agent-agnostic; reuse them so the two
# tools cannot drift apart. Importing does not touch the psmflow relabel code path.
from tools.relabel_reward_rhat import relabel_stats, scale_to_real


def select_preimage_u(aug, use_point_preimage, u_clip):
    """The per-row action latent u', clipped to +-u_clip.

    Point mode returns the stored backward-ODE preimage; mixture mode returns the weighted
    mean of the EM mixture components (a deterministic per-row draw, unlike the Dataset's
    sampled draw). Matches how psmgoal reads u at inference (agents/psmgoal.py:424).
    """
    if use_point_preimage:
        u = np.asarray(aug["noise_preimage_point"], np.float32)
    else:
        means = np.asarray(aug["noise_preimage_mean"], np.float32)      # (N, K, A)
        weights = np.asarray(aug["noise_preimage_weights"], np.float32)  # (N, K)
        u = (weights[..., None] * means).sum(1).astype(np.float32)
    return np.clip(u, -float(u_clip), float(u_clip)).astype(np.float32)


def rhat_goal_set(agent, next_observations, u_rows, goals, w, batch_size=20000):
    """r_hat(s', u') = mean over the goal axis of mesh_M(s', u', G, w*), row by row.

    Exactly the goal-averaged measure select_latent scans at acting time
    (agents/psmgoal.py:360), evaluated at the recorded preimage u' rather than prior draws.
    """
    import jax.numpy as jnp
    goals = jnp.asarray(goals, jnp.float32)
    w = jnp.asarray(w, jnp.float32)
    out = np.empty((len(next_observations),), np.float32)
    for lo in range(0, len(next_observations), batch_size):
        obs = jnp.asarray(next_observations[lo:lo + batch_size], jnp.float32)
        u = jnp.asarray(u_rows[lo:lo + batch_size], jnp.float32)
        w_rows = jnp.broadcast_to(w, (obs.shape[0], w.shape[0]))
        m = agent.mesh_M(obs, u, goals, w_rows)                 # (B, G)
        out[lo:lo + batch_size] = np.asarray(m.mean(axis=1), np.float32)
    return out


def _projected_phi(agent, obs, k_u, k_g, proj_seed, u_clip, goal_states, batch_size=20000):
    """f(s) from PsmgoalProjectedPhi with the restored RLUMeasure basis: goal/action-
    marginalised, sphere-normalised state feature. Returns (N, z_dim) numpy."""
    import jax.numpy as jnp

    from agents.f_psmflow import _sample_projection_u
    from utils.psm_networks import PsmgoalProjectedPhi

    z_dim = int(agent.config["z_dim"])
    mcfg = agent.config["measure"]
    proj_u = _sample_projection_u(proj_seed, k_u, int(agent.config["action_dim"]), u_clip)
    proj_g = jnp.asarray(goal_states[:k_g], jnp.float32)
    phi_def = PsmgoalProjectedPhi(
        z_dim=z_dim, measure_hidden_dim=int(mcfg["hidden_dim"]),
        measure_hidden_layers=int(mcfg["hidden_layers"]), U=proj_u, G=proj_g)
    params = {"params": {"measure": agent.basis.params}}
    out = np.empty((len(obs), z_dim), np.float32)
    for lo in range(0, len(obs), batch_size):
        f = phi_def.apply(params, jnp.asarray(obs[lo:lo + batch_size], jnp.float32))
        out[lo:lo + batch_size] = np.asarray(f, np.float32)
    return out


def rhat_regression(agent, obs, real_reward, k_u, k_g, proj_seed, u_clip, goal_states):
    """Control: r_hat = f(s)^T w with w = lstsq(f, real_reward). Returns (r_hat, w, f)."""
    f = _projected_phi(agent, obs, k_u, k_g, proj_seed, u_clip, goal_states)
    w, *_ = np.linalg.lstsq(f, np.asarray(real_reward, np.float64), rcond=None)
    r_hat = (f @ w).astype(np.float32)
    return r_hat, w.astype(np.float32), f


def write_reward_override(out, r_out, meta):
    """Write the reward_override npz (`rewards`, float32, N) and its sibling meta.json."""
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, rewards=np.asarray(r_out, np.float32))
    with open(out + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="psmgoal run dir (glob ok, one match)")
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--env", default="cube-single-play-singletask-v0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--readout", choices=["goal_set", "regression"], default="goal_set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--relabel_size", type=int, default=10000)
    ap.add_argument("--reward_shift", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="relabel only the first N rows")
    ap.add_argument("--match_real_scale", action="store_true",
                    help="write scale * r_hat + offset - reward_shift (least-squares fit to "
                         "the shifted reward, shifted back) instead of raw r_hat")
    # regression control knobs (marginalisation mesh for f(s))
    ap.add_argument("--reg_k_u", type=int, default=8)
    ap.add_argument("--reg_k_g", type=int, default=32)
    ap.add_argument("--reg_proj_seed", type=int, default=0)
    a = ap.parse_args()

    import ml_collections

    from agents import agents
    from agents.psmgoal import get_config
    from envs.env_utils import make_env_and_datasets
    from main import _lists_to_tuples
    from tools.eval_checkpoint import _cli_agent_keys, merge_run_config
    from utils.datasets import Dataset
    from utils.flax_utils import restore_agent
    from utils.flow_inversion import load_augmented_dataset, repair_invalid_preimages

    runs = glob.glob(a.run_dir)
    assert len(runs) == 1, f"--run_dir matched {len(runs)}: {runs}"
    run_dir = runs[0].rstrip("/")

    # Agent config: the run's flags.json is the source of truth (flow paths, preimage path,
    # net widths, u_clip, coef_source), inherited onto psmgoal's default schema.
    base = json.loads(json.dumps(get_config().to_dict()))
    merged, prov = merge_run_config({"agent_name": "psmgoal", **base}, run_dir, _cli_agent_keys())
    cfg = ml_collections.ConfigDict(_lists_to_tuples(merged))
    flow_ckpt_path = cfg["flow_ckpt_path"]
    flow_ckpt_epoch = int(cfg["flow_ckpt_epoch"])
    preimage_path = cfg["preimage_path"]
    use_point = bool(cfg["use_point_preimage"])
    u_clip = float(cfg["u_clip"])
    assert flow_ckpt_path and preimage_path, (
        f"{run_dir}/flags.json is missing flow_ckpt_path or preimage_path")

    # Dataset + preimages, spliced the way tools/eval_checkpoint.py serves them to psmgoal.
    np.random.seed(a.seed)
    _, _, train_dataset, _ = make_env_and_datasets(a.env, frame_stack=None)
    train_dataset = dict(train_dataset)
    aug, _ = repair_invalid_preimages(load_augmented_dataset(preimage_path))
    assert aug["observations"].shape[0] == train_dataset["observations"].shape[0], (
        "preimage npz row count != dataset (wrong env or stale npz)")
    for k in aug:
        if k.startswith("noise_preimage"):
            train_dataset[k] = aug[k]
    ds = Dataset.create(**train_dataset)
    ds.return_preimage_noise = True
    ds.preimage_point_mode = use_point
    ex = ds.sample(1)

    agent = agents["psmgoal"].create(a.seed, ex["observations"], ex["actions"], cfg)
    agent = restore_agent(agent, run_dir, a.epoch)

    # Goal set G and coefficient w*, exactly as the eval builds them: the relabel batch is both
    # the goal source (its rewarding next states) and the (s, u) sample the coefficient uses.
    n_relabel = min(ds.size, int(a.relabel_size))
    zb = ds.sample(n_relabel)
    agent = agent.infer_eval_goals(zb, zb["rewards"] + a.reward_shift)

    n = ds.size if a.limit <= 0 else min(ds.size, int(a.limit))
    next_obs = np.asarray(ds["next_observations"][:n], np.float32)
    r_true = np.asarray(ds["rewards"][:n], np.float32)

    if a.readout == "goal_set":
        u_all = select_preimage_u(aug, use_point, u_clip)[:n]
        r_hat = rhat_goal_set(agent, next_obs, u_all, agent.eval_goals, agent.eval_w_star)
        rhat_desc = ("r_hat = mean_g M(next_observations, u', g) over the eval goal set "
                     "(u' = clipped flow preimage)")
    else:
        goal_states = np.asarray(ds["observations"], np.float32)
        r_hat, _, _ = rhat_regression(agent, next_obs, r_true + a.reward_shift,
                                      a.reg_k_u, a.reg_k_g, a.reg_proj_seed, u_clip,
                                      goal_states)
        rhat_desc = ("r_hat = f(next_observations)^T w, w = lstsq(f, shifted reward), "
                     "f = PsmgoalProjectedPhi (goal/action-marginalised state feature)")

    assert np.isfinite(r_hat).all(), "non-finite r_hat"
    stats = relabel_stats(r_hat, r_true, a.reward_shift)

    r_out, output = r_hat, {"kind": "raw", "reward": rhat_desc}
    if a.match_real_scale:
        r_out, scale, offset = scale_to_real(r_hat, r_true, a.reward_shift)
        d_o, d_t = r_out - r_out.mean(), r_true - r_true.mean()
        output = {"kind": "scaled", "reward": "scale * r_hat + offset - reward_shift",
                  "affine_scale": scale, "affine_offset": offset,
                  "shift_back": float(a.reward_shift),
                  "mean": float(r_out.mean()), "std": float(r_out.std()),
                  "min": float(r_out.min()), "max": float(r_out.max()),
                  "corr_to_real": float((d_o * d_t).mean() / (d_o.std() * d_t.std() + 1e-12))}

    r_shifted = r_true + a.reward_shift
    meta = {"kind": "reward_override", "agent": "psmgoal", "readout": a.readout,
            "output": output, "rhat": rhat_desc,
            "env_name": a.env, "num_rows": int(n), "limit": int(a.limit),
            "dataset_rows": int(ds.size),
            "rewarding_frac": float((r_shifted > 0.5).mean()),
            "checkpoint": {"run_dir": run_dir, "epoch": int(a.epoch),
                           "flags_json": prov["flags_json"],
                           "flow_ckpt_path": flow_ckpt_path,
                           "flow_ckpt_epoch": int(flow_ckpt_epoch),
                           "preimage_path": preimage_path},
            "readout_settings": {"seed": int(a.seed), "relabel_size": int(n_relabel),
                                 "reward_shift": float(a.reward_shift),
                                 "use_point_preimage": use_point, "u_clip": u_clip,
                                 "k_goals": int(cfg["k_goals"]),
                                 "coef_source": str(cfg["coef_source"]),
                                 "eval_w_star_norm": float(np.linalg.norm(agent.eval_w_star)),
                                 "reg_k_u": int(a.reg_k_u), "reg_k_g": int(a.reg_k_g),
                                 "reg_proj_seed": int(a.reg_proj_seed)},
            "fit_on_all_rows": stats}
    write_reward_override(a.out, r_out, meta)

    print(f"wrote {a.out} ({n} rows, {a.readout}, {output['kind']}) and .meta.json")
    print(f"raw r_hat mean/std {stats['rhat_mean']:.4f}/{stats['rhat_std']:.4f}  "
          f"min/max {stats['rhat_min']:.4f}/{stats['rhat_max']:.4f}  "
          f"corr {stats['corr']:.4f}  top-{stats['top_frac']:.0%} precision "
          f"{stats['top_precision']:.4f} (base {stats['base_rate']:.4f})")
    print(f"rewarding rows (shifted>0.5): {meta['rewarding_frac']:.4f}  "
          f"eval_w_star norm {meta['readout_settings']['eval_w_star_norm']:.4f}")
    if a.match_real_scale:
        print(f"scaled: affine (a, b) = ({output['affine_scale']:.6f}, {output['affine_offset']:.6f})  "
              f"mean/std {output['mean']:.4f}/{output['std']:.4f}  "
              f"min/max {output['min']:.4f}/{output['max']:.4f}  "
              f"corr to real {output['corr_to_real']:.4f}")


if __name__ == "__main__":
    main()
