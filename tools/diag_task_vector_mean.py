"""Is the inferred task vector the GOAL direction, or just the mean feature?

`infer_z` uses w = project(E_D[(r + shift) phi(s')]). On a task where only ~2% of rows pay,
the shifted reward is 0 almost everywhere and 1 on the payers, so

    E[(r+1) phi] = p_goal * E[phi | goal]
                = p_goal * mu + p_goal * (E[phi | goal] - mu),  mu = E[phi]

Both terms carry p_goal. Sparse rewards alone therefore do not imply that a standalone
mean-feature term dominates the readout. Centering gives the reward-feature covariance

    w_c = E[(r+1)(phi - mu)] = p_goal * (E[phi | goal] - mu)

which is independent of an additive reward shift when the same sample defines mu. Using
this readout with an approximate, uncentered successor model can change action rankings;
better control performance does not follow from centering algebra alone.

This reports |mu|, cos(w, mu), cos(w_c, mu), cos(w, w_c), and Gram eigenvalues. A large
Gram eigenvalue can reflect either a mean direction or covariance anisotropy; these
quantities do not establish a cause of policy failure. Eval-only; no weights change.
"""
import argparse
import json
import os

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated <label>=<dir>@<epoch>")
    ap.add_argument("--flow", required=True)
    ap.add_argument("--flow_epoch", type=int, default=500000)
    ap.add_argument("--preimages", default="")
    ap.add_argument("--env", required=True)
    ap.add_argument("--relabel", type=int, default=10000)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report_out", required=True)
    a = ap.parse_args()

    import jax.numpy as jnp

    from agents import agents
    from agents.psmflow import get_config
    from envs.env_utils import make_env_and_datasets
    from tools.eval_checkpoint import merge_run_config
    from utils.datasets import Dataset
    from utils.flax_utils import restore_agent

    _, _, train_dataset, _ = make_env_and_datasets(a.env, frame_stack=None, add_info=True)
    ds = Dataset.create(**train_dataset)
    ex = ds.sample(1)
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(ds.size, min(a.relabel, ds.size), replace=False))
    nxt = jnp.asarray(np.asarray(ds["next_observations"][idx], np.float32))
    rew = np.asarray(ds["rewards"][idx], np.float64).ravel() + a.shift

    def cos(x, y):
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))

    out = {"probe": "is the inferred task vector the goal direction or the mean feature",
           "env": a.env, "relabel_rows": len(idx), "shift": a.shift,
           "p_goal": float((rew > rew.min()).mean()), "runs": {}}

    for spec in a.runs.split(","):
        label, rest = spec.split("=", 1)
        run_dir, epoch = rest.rsplit("@", 1)
        base = json.loads(json.dumps(get_config().to_dict()))
        cfg, _ = merge_run_config(base, run_dir, {})
        cfg["flow_ckpt_path"], cfg["flow_ckpt_epoch"] = a.flow, a.flow_epoch
        cfg["preimage_path"] = a.preimages or None
        agent = agents["psmflow"].create(a.seed, ex["observations"], ex["actions"], cfg)
        agent = restore_agent(agent, run_dir, int(epoch))

        ph = np.asarray(agent.phi(nxt), np.float64)              # (N, z), as infer_z reads it
        mu = ph.mean(0)
        w = (rew[None] @ ph).ravel() / len(rew)                  # E[(r+shift) phi]
        w_c = (rew[None] @ (ph - mu[None])).ravel() / len(rew)   # centred
        gram = (ph.T @ ph) / len(ph)
        eig = np.linalg.eigvalsh(gram)

        rec = {"run_dir": run_dir, "epoch": int(epoch),
               "phi_norm_mean": float(np.linalg.norm(ph, axis=-1).mean()),
               "mu_norm": float(np.linalg.norm(mu)),
               "w_norm": float(np.linalg.norm(w)),
               "w_c_norm": float(np.linalg.norm(w_c)),
               "cos_w_mu": cos(w, mu), "cos_wc_mu": cos(w_c, mu), "cos_w_wc": cos(w, w_c),
               "gram_eig_max": float(eig[-1]), "gram_eig_min": float(eig[0])}
        out["runs"][label] = rec
        print(f"{label:26s} |mu|={rec['mu_norm']:6.3f}  cos(w,mu)={rec['cos_w_mu']:+.3f}  "
              f"cos(w_c,mu)={rec['cos_wc_mu']:+.3f}  cos(w,w_c)={rec['cos_w_wc']:+.3f}  "
              f"eig_max={rec['gram_eig_max']:6.3f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.report_out)), exist_ok=True)
    with open(a.report_out, "w") as f:
        json.dump(out, f, indent=2)
    print("report ->", a.report_out)


if __name__ == "__main__":
    main()
