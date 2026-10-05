"""Is psmgoal's phi orthonormal? Pre-check for an auxiliary orthogonality loss (2026-09-22).

The Lagrangian inference (coef_source=lp) maximises a LINEAR objective in w on the sqrt(D)
sphere, so when its constraint is inactive it returns the direction of the objective's
gradient. That direction equals the least-squares reward fit only when the feature Gram is
the identity. This tool measures, on a restored psmgoal checkpoint:

  mesh Gram   G = mean_ij phi(s_i,u_i,s+_j) phi(s_i,u_i,s+_j)^T   over a training-style mesh
              (s_i, u_i) from the dataset, s+_j = the batch's next states (the TD loss's mesh)
  feature Gram  of Phi[i] = mean_{g in G} phi(s'_i, u_i, g), the feature coef_source=regression
              regresses the reward on (G = rewarding next states, as infer_eval_goals builds it)
  directions  a_lp  = mean_i mean_g phi(s_i, u_i, g)   (gradient of the lp objective)
              a_rphi = mean_i Phi[i] r_i                (E[phi r] on the regression feature)
              w_ls  = lstsq(Phi, r)                     (what coef_source=regression uses)
              and the cosines between them.

Reported per Gram: eigenvalue min/max, condition number, trace/D, ||G - I||_F / sqrt(D).
Output is JSON via --report_out.

Run (CPU is fine):
  JAX_PLATFORMS=cpu OGBENCH_DATASET_DIR=... .venv/bin/python tools/diag_psmgoal_gram.py \
      --run_dir '$PSM_DATA/exp/PSMFLows/psmgoal_lift_cube_gc/sd000_*' --epoch 750000 \
      --report_out $PSM_DATA/logs/diag_psmgoal_gram_sd000_750k.json
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax


def gram_stats(G):
    D = G.shape[0]
    ev = np.linalg.eigvalsh(G)
    evp = np.clip(ev, 0.0, None)
    return {"eig_min": float(ev[0]), "eig_max": float(ev[-1]),
            "cond": float(ev[-1] / max(ev[0], 1e-12)), "trace_over_D": float(np.trace(G) / D),
            "dev_from_I": float(np.linalg.norm(G - np.eye(D)) / np.sqrt(D)),
            "top1_frac": float(evp[-1] / evp.sum()), "top5_eigs": [float(x) for x in ev[-5:][::-1]],
            # participation ratio: D for an isotropic Gram, 1 for a rank-one Gram
            "effective_rank": float(evp.sum() ** 2 / (evp ** 2).sum()),
            "n_eig_above_1pct_max": int((evp > 0.01 * evp[-1]).sum())}


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="psmgoal run dir (glob ok, one match)")
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--env", default="cube-single-play-singletask-v0")
    ap.add_argument("--report_out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mesh_n", type=int, default=256, help="rows per mesh (TD batch size)")
    ap.add_argument("--mesh_batches", type=int, default=8)
    ap.add_argument("--feat_rows", type=int, default=20000, help="rows for the regression feature")
    ap.add_argument("--k_goals", type=int, default=32)
    ap.add_argument("--reward_shift", type=float, default=1.0)
    a = ap.parse_args()

    import jax.numpy as jnp
    import ml_collections

    from agents import agents
    from agents.psmgoal import REWARDING_THRESHOLD, get_config
    from envs.env_utils import make_env_and_datasets
    from main import _lists_to_tuples
    from tools.eval_checkpoint import _cli_agent_keys, merge_run_config
    from tools.relabel_reward_rhat_psmgoal import select_preimage_u
    from utils.datasets import Dataset
    from utils.flax_utils import restore_agent
    from utils.flow_inversion import load_augmented_dataset, repair_invalid_preimages

    runs = glob.glob(a.run_dir)
    assert len(runs) == 1, f"--run_dir matched {len(runs)}: {runs}"
    run_dir = runs[0].rstrip("/")
    base = json.loads(json.dumps(get_config().to_dict()))
    merged, prov = merge_run_config({"agent_name": "psmgoal", **base}, run_dir, _cli_agent_keys())
    cfg = ml_collections.ConfigDict(_lists_to_tuples(merged))
    use_point, u_clip = bool(cfg["use_point_preimage"]), float(cfg["u_clip"])

    np.random.seed(a.seed)
    _, _, train_dataset, _ = make_env_and_datasets(a.env, frame_stack=None)
    train_dataset = dict(train_dataset)
    aug, _ = repair_invalid_preimages(load_augmented_dataset(cfg["preimage_path"]))
    assert aug["observations"].shape[0] == train_dataset["observations"].shape[0]
    for k in aug:
        if k.startswith("noise_preimage"):
            train_dataset[k] = aug[k]
    ds = Dataset.create(**train_dataset)
    ds.return_preimage_noise = True
    ds.preimage_point_mode = use_point
    ex = ds.sample(1)
    agent = agents["psmgoal"].create(a.seed, ex["observations"], ex["actions"], cfg)
    agent = restore_agent(agent, run_dir, a.epoch)
    D = int(cfg["z_dim"])

    obs_all = np.asarray(ds["observations"], np.float32)
    next_all = np.asarray(ds["next_observations"], np.float32)
    r_all = np.asarray(ds["rewards"], np.float32) + a.reward_shift
    u_all = select_preimage_u(aug, use_point, u_clip)
    rng = np.random.default_rng(a.seed)

    def phi_of(o, u, g):
        p, _ = agent.basis(jnp.asarray(o), jnp.asarray(u), jnp.asarray(g), params=agent.basis.params)
        return np.asarray(p, np.float64)

    # Mesh Gram: the TD loss's (s_i,u_i) x s+_j mesh, s+_j = the batch's next states.
    G_mesh = np.zeros((D, D))
    for _ in range(a.mesh_batches):
        idx = rng.choice(obs_all.shape[0], a.mesh_n, replace=False)
        o, u, g = obs_all[idx], u_all[idx], next_all[idx]
        N = a.mesh_n
        oo = np.repeat(o, N, 0)
        uu = np.repeat(u, N, 0)
        gg = np.tile(g, (N, 1))
        p = phi_of(oo, uu, gg)
        G_mesh += p.T @ p / p.shape[0]
    G_mesh /= a.mesh_batches

    # Goal set G: rewarding next states, as infer_eval_goals builds it.
    rew_rows = np.nonzero(r_all > REWARDING_THRESHOLD)[0]
    goals = next_all[rng.choice(rew_rows, a.k_goals, replace=rew_rows.size < a.k_goals)]
    Kg = goals.shape[0]
    idx = rng.choice(obs_all.shape[0], a.feat_rows, replace=False)
    # Oversample rewarding rows so the regression has signal (the sparse reward is ~1-2%).
    idx = np.concatenate([idx, rng.choice(rew_rows, min(a.feat_rows // 10, rew_rows.size), replace=False)])

    def goal_avg(o, u, chunk=2000):
        out = []
        for s in range(0, o.shape[0], chunk):
            oc, uc = o[s:s + chunk], u[s:s + chunk]
            n = oc.shape[0]
            p = phi_of(np.repeat(oc, Kg, 0), np.repeat(uc, Kg, 0), np.tile(goals, (n, 1)))
            out.append(p.reshape(n, Kg, D).mean(1))
        return np.concatenate(out)

    Phi = goal_avg(next_all[idx], u_all[idx])            # regression feature at s'
    r = r_all[idx]
    A_lp = goal_avg(obs_all[idx], u_all[idx])            # lp objective feature at s
    G_feat = Phi.T @ Phi / Phi.shape[0]
    w_ls = np.linalg.lstsq(Phi, r, rcond=None)[0]
    a_rphi = (Phi * r[:, None]).mean(0)
    a_lp = A_lp.mean(0)
    w_white = np.linalg.solve(G_feat + 1e-8 * np.eye(D), a_rphi)

    report = {
        "run_dir": run_dir, "epoch": a.epoch, "env": a.env, "z_dim": D,
        "flags_json": prov.get("flags_json"),
        "mesh": {"n": a.mesh_n, "batches": a.mesh_batches, **gram_stats(G_mesh)},
        "regression_feature": {"rows": int(Phi.shape[0]), "k_goals": Kg, **gram_stats(G_feat)},
        "cos": {"a_lp_vs_w_ls": cos(a_lp, w_ls), "a_rphi_vs_w_ls": cos(a_rphi, w_ls),
                "a_lp_vs_a_rphi": cos(a_lp, a_rphi), "whitened_a_rphi_vs_w_ls": cos(w_white, w_ls)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.report_out)), exist_ok=True)
    with open(a.report_out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("report ->", a.report_out)


if __name__ == "__main__":
    main()
