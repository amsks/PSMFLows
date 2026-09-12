"""Can the frozen basis phi express the task reward at all, linearly?

Every zero-shot critic in this project reads a reward through `Q = psi^T w` with
`w = E_D[r phi(x)]` (Cor. `reward-inference`). If the best LINEAR read-out of phi cannot
reconstruct the reward, then no psi on that phi can represent the task, and work on the
measure is capped by the basis regardless of how well the Bellman side behaves.

The gate, 2026-09-09, agreed with the oversight session before running:

  R^2 above 0.5  the linear readout carries the signal; the Bellman route is viable and the
                 zero-shot arms are worth pushing.
  R^2 below 0.2  phi cannot express the reward; every zero-shot critic on this phi is
                 capped, and the work redirects to phi rather than to psi.

Four fits per checkpoint, because the agent's estimator is not the topline:

  ls            least squares of r on phi(s'), no intercept -- the TOPLINE, the best any
                linear readout of this phi can do
  ls_intercept  the same with an intercept, which the agent has no way to represent; the
                gap between the two is what the missing constant costs
  closed_form   w = E[r phi], the estimator `infer_z` actually uses, unnormalised
  closed_norm   the same projected to the unit sphere, which is what `norm_z=True` deploys.
                Its raw R^2 is meaningless (the projection throws the scale away and Q is
                then defined only up to that scale), so it is also reported after an optimal
                scalar rescale, which isolates the DIRECTION of w from its magnitude.

Both reward conventions are reported: the raw dataset reward (-1/0 on cube) and the shifted
reward the eval path uses (`eval_reward_shift=1.0`, giving 0/1). They differ by a constant,
which a no-intercept linear model cannot absorb, so they do not give the same R^2 -- that is
the point of reporting both.

Residuals are split by success rows (r > min) against the rest: a fit can look adequate in
aggregate while being blind on the 2% of rows that carry all the signal.

Run (CPU; ~1 min per checkpoint):
  .venv/bin/python tools/diag_reward_readout.py \
      --runs $PSM_DATA/exp/PSMFLows/affine_strict_cube \
      --flow $PSM_DATA/flow/cube-single-play \
      --preimages $PSM_DATA/preimages/cube-single-play.npz \
      --out $PSM_DATA/logs/diag_reward_readout_cube.json
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax


def _r2(pred, y):
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def _fits(phi, r):
    """The four readouts, plus the residual split by success rows."""
    hot = r > r.min()
    out = {"n": len(r), "frac_success_rows": float(hot.mean()),
           "reward_mean": float(r.mean()), "reward_std": float(r.std())}

    w_ls, *_ = np.linalg.lstsq(phi, r, rcond=None)
    p_ls = phi @ w_ls
    out["ls"] = {"r2": _r2(p_ls, r), "w_norm": float(np.linalg.norm(w_ls))}
    # HELD OUT. The in-sample least-squares R^2 is what a rank-deficient basis can fit, not
    # what it can predict: `lstsq` puts unbounded weight on directions where phi has almost
    # no variance, and on pointmaze the Gram's smallest eigenvalue is 2e-4. Fit on the first
    # half, score on the second. A large in-sample/held-out gap means the topline was
    # overfitting and must not be read as "what a usable readout could achieve".
    h = len(r) // 2
    w_h, *_ = np.linalg.lstsq(phi[:h], r[:h], rcond=None)
    out["ls"]["r2_heldout"] = _r2(phi[h:] @ w_h, r[h:])
    out["ls"]["r2_insample_half"] = _r2(phi[:h] @ w_h, r[:h])
    out["ls"]["w_norm_half"] = float(np.linalg.norm(w_h))

    ones = np.ones((len(r), 1), phi.dtype)
    w_i, *_ = np.linalg.lstsq(np.concatenate([phi, ones], 1), r, rcond=None)
    p_i = np.concatenate([phi, ones], 1) @ w_i
    out["ls_intercept"] = {"r2": _r2(p_i, r), "intercept": float(w_i[-1])}

    w_cf = (r[None] @ phi).ravel() / len(r)          # E[r phi], `infer_z` before projection
    p_cf = phi @ w_cf
    # Optimal rescale isolates the DIRECTION of w from its magnitude; `norm_z=True` throws
    # the magnitude away at eval anyway, and Q's scale does not change an argmax.
    a = float((p_cf @ r) / (p_cf @ p_cf + 1e-12))
    out["closed_form"] = {"r2": _r2(p_cf, r), "r2_rescaled": _r2(a * p_cf, r),
                          "rescale": a, "w_norm": float(np.linalg.norm(w_cf)),
                          "cos_to_ls": float(w_cf @ w_ls /
                                             (np.linalg.norm(w_cf) * np.linalg.norm(w_ls) + 1e-12))}

    for name, pred in (("ls", p_ls), ("closed_form", a * p_cf)):
        res = r - pred
        out[name]["resid_success_mean"] = float(res[hot].mean()) if hot.any() else None
        out[name]["resid_success_absmean"] = float(np.abs(res[hot]).mean()) if hot.any() else None
        out[name]["resid_other_absmean"] = float(np.abs(res[~hot]).mean())
        out[name]["pred_success_mean"] = float(pred[hot].mean()) if hot.any() else None
        out[name]["pred_other_mean"] = float(pred[~hot].mean())
        # The quantity a critic needs: does the readout SEPARATE rewarding rows from the
        # rest? A low R^2 with clean separation is a scale problem, not a signal problem.
        if hot.any():
            sep = (pred[hot].mean() - pred[~hot].mean()) / (pred.std() + 1e-12)
            out[name]["separation_in_sds"] = float(sep)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--flow", required=True)
    ap.add_argument("--flow_epoch", type=int, default=500000)
    ap.add_argument("--preimages", default="")
    ap.add_argument("--epochs", default="250000,350000,450000,500000")
    ap.add_argument("--rows", type=int, default=200000)
    ap.add_argument("--env", default="cube-single-play-singletask-v0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import jax.numpy as jnp
    import ml_collections

    from agents import agents
    from agents.psmflow import get_config
    from envs.env_utils import make_env_and_datasets
    from main import _lists_to_tuples
    from tools.eval_checkpoint import _cli_agent_keys, merge_run_config
    from utils.datasets import Dataset
    from utils.flax_utils import restore_agent

    np.random.seed(a.seed)
    _, _, train_dataset, _ = make_env_and_datasets(a.env, frame_stack=None, add_info=True)
    ds = Dataset.create(**train_dataset)
    ex = ds.sample(1)
    idx = np.sort(np.random.default_rng(a.seed).choice(ds.size, min(a.rows, ds.size),
                                                       replace=False))
    next_obs = np.asarray(ds["next_observations"][idx], np.float32)
    r_raw = np.asarray(ds["rewards"][idx], np.float64).ravel()

    base = json.loads(json.dumps(get_config().to_dict()))
    base["flow_ckpt_path"], base["flow_ckpt_epoch"] = a.flow, a.flow_epoch
    base["preimage_path"] = a.preimages or None

    out = {"probe": "linear reconstruction of the task reward from the frozen basis phi",
           "env": a.env, "n_rows": len(idx), "z_dim": None,
           "gate": {"viable_above": 0.5, "capped_below": 0.2,
                    "note": "R^2 of the TOPLINE (`ls`) decides the gate; the closed form "
                            "is what the agent deploys and cannot beat it"},
           "checkpoints": {}}

    for run_dir in sorted(glob.glob(os.path.join(a.runs, "*/"))):
        run_dir = run_dir.rstrip("/")
        sd = os.path.basename(run_dir)[:5]
        for epoch in [int(x) for x in a.epochs.split(",")]:
            if not os.path.isfile(os.path.join(run_dir, f"params_{epoch}.pkl")):
                continue
            merged, _ = merge_run_config({"agent_name": "psmflow", **base},
                                         run_dir, _cli_agent_keys())
            cfg = ml_collections.ConfigDict(_lists_to_tuples(merged))
            agent = agents["psmflow"].create(a.seed, ex["observations"], ex["actions"], cfg)
            agent = restore_agent(agent, run_dir, epoch)
            out["z_dim"] = int(cfg["z_dim"])

            phi = np.concatenate([
                np.asarray(agent.phi(jnp.asarray(next_obs[i:i + 20000])), np.float64)
                for i in range(0, len(next_obs), 20000)])
            # The closed form w = E[r phi] is the least-squares solution ONLY when
            # E[phi phi^T] = I, which is exactly what the ortho term is there to enforce.
            # When the Gram is far from identity the deployed estimator is not merely
            # suboptimal, it can point somewhere else entirely -- so its deviation is the
            # mechanism behind any gap between `closed_form` and `ls`, and is measured here
            # rather than inferred.
            gram = (phi.T @ phi) / len(phi)
            ev = np.linalg.eigvalsh(gram)
            eye = np.eye(gram.shape[0])
            rec = {"phi_norm_mean": float(np.linalg.norm(phi, axis=-1).mean()),
                   "gram": {"eig_min": float(ev.min()), "eig_max": float(ev.max()),
                            "eig_mean": float(ev.mean()),
                            "cond": float(ev.max() / max(ev.min(), 1e-12)),
                            "rel_dev_from_I": float(np.linalg.norm(gram - eye) /
                                                    np.linalg.norm(eye)),
                            "offdiag_absmean": float(np.abs(gram - np.diag(np.diag(gram))).mean())},
                   # Carried so the cross-ARM table is self-describing: the 2026-09-09
                   # extension asks whether the cap moves with the orthonormality weight.
                   "ortho_coef": float(cfg["ortho_coef"]),
                   "ortho_mode": str(cfg["ortho_mode"]),
                   "lr_sf": float(cfg["lr_sf"]),
                   "z_dim": int(cfg["z_dim"]),
                   "discount": float(cfg["discount"]),
                   "raw_reward": _fits(phi, r_raw),
                   "shifted_reward": _fits(phi, r_raw + 1.0)}
            out["checkpoints"][f"{sd}@{epoch}"] = rec
            print(f"{sd}@{epoch:>7d}  R2 topline raw {rec['raw_reward']['ls']['r2']:.4f}  "
                  f"shifted {rec['shifted_reward']['ls']['r2']:.4f}  |  closed-form rescaled "
                  f"{rec['shifted_reward']['closed_form']['r2_rescaled']:.4f}  |  "
                  f"sep {rec['shifted_reward']['ls']['separation_in_sds']:.2f}sd  |  "
                  f"gram dev {rec['gram']['rel_dev_from_I']:.3f} cond {rec['gram']['cond']:.1f}",
                  flush=True)

    tops = [v["shifted_reward"]["ls"]["r2"] for v in out["checkpoints"].values()]
    if tops:
        out["topline_r2_shifted"] = {"mean": float(np.mean(tops)), "min": float(np.min(tops)),
                                     "max": float(np.max(tops))}
        m = float(np.mean(tops))
        out["verdict"] = ("VIABLE: the linear readout carries the reward" if m > 0.5 else
                          "CAPPED: phi cannot express the reward; the work is on phi, not psi"
                          if m < 0.2 else
                          "AMBIGUOUS: between the pre-registered thresholds")
        print(f"\nmean topline R^2 (shifted) = {m:.4f}  ->  {out['verdict']}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"report -> {a.out}")


if __name__ == "__main__":
    main()
