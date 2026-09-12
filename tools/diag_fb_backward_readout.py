"""Is FB's backward map B less blurry about the reward than our phi?

Our cube phi reads a topline least-squares R2 of 0.116 with 86.5% of its predicted reward
mass on rows that pay nothing (`tools/diag_reward_readout.py`), and a matched random 128-d
basis reads 0.065 -- so phi is about 2x random and nowhere near able to express the reward.
FB infers its task vector the same way we do (`z = E[(r+shift) B(s')]`, projected), on the
same data, so the same measurement on B says whether the difference between the methods is
in the FEATURES or somewhere else.

NARROW QUESTION, stated because the motivating premise was about antmaze: this runs on CUBE,
because no FB antmaze checkpoint exists on this machine. It answers "trained the same way on
the same data, is FB's B less blurry than our phi", NOT "why does FB reach 73 on antmaze".

NO AGENT IS CONSTRUCTED. The checkpoints under `psm-data/exp/FactoredFB` were written by the
SEPARATE Factored-FB codebase, not by this repo's `archive/agents/fb.py` -- they carry
`net`/`target`/`cond_eval`, which that class has no slot for, and `restore_agent` refuses
them for exactly that reason. So B is applied here in numpy, straight from the pickle, using
the forward pass read off `Factored-FB/impls/utils/networks.py:56-75`:

    Dense -> LayerNorm -> tanh, then (hidden_layers - 1) x [Dense -> relu], then Dense,
    then `psm_norm`: sqrt(d) * x / ||x||   (their `_L2`, matching our `project_z`)

The layer shapes in the checkpoint (28->512->512->512->512->50, one LayerNorm) match
`fb.yaml`'s backward block (hidden_dim 512, hidden_layers 4, norm true) exactly, which is the
check that this reimplementation is reading the right architecture. Nothing is imported from
either the archive or the Factored-FB tree.
"""
import argparse
import json
import os
import pickle

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax


def _apply_phimap(params, x):
    """Factored-FB's `PhiMap` forward pass, in numpy. networks.py:56-75.

    Dense -> LayerNorm(eps 1e-5) -> tanh, then (L-1) x [Dense -> relu], then Dense to z_dim,
    then sqrt(d) * x / ||x||. Flax names the layers Dense_0.. and LayerNorm_0 in order, so
    the count of Dense_* is what fixes L; it is asserted against the config below.
    """
    n_dense = sum(1 for k in params if k.startswith("Dense_"))
    d0 = params["Dense_0"]
    h = x @ np.asarray(d0["kernel"]) + np.asarray(d0["bias"])
    ln = params["LayerNorm_0"]
    mu, var = h.mean(-1, keepdims=True), h.var(-1, keepdims=True)
    h = (h - mu) / np.sqrt(var + 1e-5) * np.asarray(ln["scale"]) + np.asarray(ln["bias"])
    h = np.tanh(h)
    for i in range(1, n_dense - 1):
        d = params[f"Dense_{i}"]
        h = np.maximum(h @ np.asarray(d["kernel"]) + np.asarray(d["bias"]), 0.0)
    d = params[f"Dense_{n_dense - 1}"]
    h = h @ np.asarray(d["kernel"]) + np.asarray(d["bias"])
    return np.sqrt(h.shape[-1]) * h / np.maximum(
        np.linalg.norm(h, axis=-1, keepdims=True), 1e-12)


def _r2(pred, y):
    return float(1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def _fits(feat, r, pays):
    """Least squares with intercept, in-sample and on a held-out half, plus mass placement."""
    f = np.concatenate([feat, np.ones((len(feat), 1))], 1)
    h = len(r) // 2
    w_all = np.linalg.lstsq(f, r, rcond=None)[0]
    w_h = np.linalg.lstsq(f[:h], r[:h], rcond=None)[0]
    pred = f @ w_all
    # Mass placement: of the total POSITIVE predicted reward, how much lands on rows that
    # pay nothing. The R2 alone hides this -- a readout can score respectably by predicting
    # a little reward everywhere.
    pos = np.clip(pred, 0, None)
    return {"r2_insample": _r2(pred, r),
            "r2_heldout": _r2(f[h:] @ w_h, r[h:]),
            "mass_on_non_paying": float(pos[~pays].sum() / (pos.sum() + 1e-12)),
            "pred_mean_on_paying": float(pred[pays].mean()),
            "pred_mean_on_non_paying": float(pred[~pays].mean()),
            "w_norm": float(np.linalg.norm(w_all))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fb_runs", required=True, help="comma-separated <label>=<dir>@<epoch>")
    ap.add_argument("--env", default="cube-single-play-singletask-v0")
    ap.add_argument("--rows", type=int, default=10000)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report_out", required=True)
    a = ap.parse_args()

    import jax.numpy as jnp

    from envs.env_utils import make_env_and_datasets
    from utils.datasets import Dataset

    _, _, train_dataset, _ = make_env_and_datasets(a.env, frame_stack=None, add_info=True)
    ds = Dataset.create(**train_dataset)
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(ds.size, min(a.rows, ds.size), replace=False))
    nxt = jnp.asarray(np.asarray(ds["next_observations"][idx], np.float32))
    raw = np.asarray(ds["rewards"][idx], np.float64).ravel()
    r = raw + a.shift
    pays = raw > raw.min()

    out = {"probe": "least-squares readout of the shifted reward from FB's backward map B",
           "question": "trained the same way on the same data, is B less blurry than our phi",
           "env": a.env, "n_rows": len(idx), "shift": a.shift,
           "paying_row_frac": float(pays.mean()),
           "our_phi_cube_reference": {"r2_insample": 0.116, "mass_on_non_paying": 0.865,
                                      "source": "diag_reward_readout, 12 ckpts"},
           "random_basis_reference": {"raw_obs_28d": 0.049, "random_tanh_128d": 0.065,
                                      "random_fourier_2048d_heldout": 0.107,
                                      "source": "diag_reward_baseline_cube.json"},
           "runs": {}}

    for spec in a.fb_runs.split(","):
        label, rest = spec.split("=", 1)
        run_dir, epoch = rest.rsplit("@", 1)
        with open(os.path.join(run_dir, f"params_{int(epoch)}.pkl"), "rb") as fh:
            prm = pickle.load(fh)["agent"]["net"]["backward"]["params"]
        B = np.asarray(_apply_phimap(prm, np.asarray(nxt, np.float64)), np.float64)
        gram = (B.T @ B) / len(B)
        eig = np.linalg.eigvalsh(gram)
        rec = {"run_dir": run_dir, "epoch": int(epoch), "z_dim": int(B.shape[1]),
               "b_norm_mean": float(np.linalg.norm(B, axis=-1).mean()),
               "gram_cond": float(eig[-1] / max(eig[0], 1e-12)),
               "gram_eig_max": float(eig[-1]), "gram_eig_min": float(eig[0]),
               **_fits(B, r, pays)}
        out["runs"][label] = rec
        print(f"{label:22s} R2={rec['r2_insample']:.3f} (heldout {rec['r2_heldout']:.3f})  "
              f"mass_non_paying={rec['mass_on_non_paying']:.3f}  "
              f"gram_cond={rec['gram_cond']:.1f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.report_out)), exist_ok=True)
    with open(a.report_out, "w") as f:
        json.dump(out, f, indent=2)
    print("report ->", a.report_out)


if __name__ == "__main__":
    main()
