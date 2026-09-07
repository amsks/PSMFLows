"""T1: does the stored preimage DISTRIBUTION decode to the recorded action?

The published round-trip (~1e-4) certifies the point inverse u*. Stage C's mixture arm
samples from the stored per-row Gaussian mixture instead, so this certifies what that
arm actually trains on: draw K samples per row from the stored posterior, decode all of
them through the flow at inversion-grade integration (flow_steps=100), and report the
per-row mean/min/max of ||G(s,u) - a_recorded||, aggregated over a seeded row subsample.
The point-preimage round-trip is recomputed on the same rows as the reference.

Antmaze note: the mixture arm there is published as broken (mean ESS 7.6, 6% > 20 --
alpha=20 does not transfer to d_a=8). Run this tool on it anyway if you want the decode
error quantified, but the JSON will carry the caveat; every current result uses the
point arm on antmaze.

Run (GPU):
  MUJOCO_GL=egl .venv/bin/python tools/diag_preimage_sampling_fidelity.py agent=fql \
      env_name=cube-single-play-singletask-v0 agent.flow_steps=100 \
      restore_path=<stage_a_flow_dir> restore_epoch=500000 \
      +npz=/data-local/amsks/PSMFLows/preimages_cube_single_a20_n200.npz \
      report_out=/data-local/amsks/PSMFLows/logs/diag_preimage_sampling_cube.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see module docstring)

import hydra
import jax
import ml_collections
import numpy as np
from omegaconf import OmegaConf

from agents import agents
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.datasets import Dataset
from utils.flax_utils import restore_agent
from utils.log_utils import write_report

N_ROWS = 100_000
K_SAMPLES = 16
BATCH = 2048


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    assert cfg.agent.agent_name == "fql", "run with agent=fql (Stage-A flow shapes)"
    assert int(cfg.agent.flow_steps) >= 100, "decode must be inversion-grade (>=100)"
    npz_path = str(cfg.npz)
    rng = np.random.default_rng(int(cfg.seed))

    env, eval_env, train_dataset, _ = make_env_and_datasets(
        cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    config = ml_collections.ConfigDict(
        _lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    ex = ds.sample(1)
    agent = agents["fql"].create(cfg.seed, ex["observations"], ex["actions"], config)
    agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    z = np.load(npz_path)
    n_total = z["observations"].shape[0]
    rows = np.sort(rng.choice(n_total, size=min(N_ROWS, n_total), replace=False))
    obs = np.asarray(z["observations"])[rows]
    act = np.asarray(z["actions"])[rows]
    mean = np.asarray(z["noise_preimage_mean"])[rows]      # (N, C, d)
    cov = np.asarray(z["noise_preimage_cov"])[rows]        # (N, C, d, d)
    wts = np.asarray(z["noise_preimage_weights"])[rows]    # (N, C)
    point = np.asarray(z["noise_preimage_point"])[rows]
    valid = np.asarray(z["preimage_valid"])[rows].astype(bool) \
        if "preimage_valid" in z.files else np.ones(len(rows), bool)
    d_a = act.shape[-1]

    # Draw K samples per row from the stored mixture (numpy; C is small).
    n = len(rows)
    comp = np.array([rng.choice(wts.shape[1], size=K_SAMPLES,
                                p=w / max(w.sum(), 1e-12)) for w in wts])   # (N, K)
    chol = np.linalg.cholesky(cov + 1e-9 * np.eye(d_a))                     # (N, C, d, d)
    eps = rng.standard_normal((n, K_SAMPLES, d_a))
    m_sel = np.take_along_axis(mean, comp[..., None], axis=1)               # (N, K, d)
    L_sel = np.take_along_axis(chol, comp[..., None, None], axis=1)         # (N, K, d, d)
    u = m_sel + np.einsum("nkij,nkj->nki", L_sel, eps)                      # (N, K, d)

    def decode(o, uu):
        return np.asarray(agent.compute_flow_actions(o, noises=uu))

    err = np.empty((n, K_SAMPLES), np.float64)
    err_pt = np.empty(n, np.float64)
    for lo in range(0, n, BATCH):
        hi = min(lo + BATCH, n)
        o = obs[lo:hi]
        for k in range(K_SAMPLES):
            a_hat = decode(o, u[lo:hi, k])
            err[lo:hi, k] = np.linalg.norm(a_hat - act[lo:hi], axis=-1)
        err_pt[lo:hi] = np.linalg.norm(decode(o, point[lo:hi]) - act[lo:hi], axis=-1)

    err_v = err[valid]
    # Tail mixture samples can diverge through the flow (the 07-28 NaN mechanism):
    # count them explicitly instead of letting one NaN poison the aggregates.
    nan_mask = ~np.isfinite(err_v)
    agg = lambda x: {"mean": float(np.nanmean(x)), "median": float(np.nanmedian(x)),
                     "p99": float(np.nanpercentile(x, 99)),
                     "max": float(np.nanmax(x))}
    err_v = np.where(nan_mask, np.nan, err_v)
    report = {
        "frac_nan_sample_errors": float(nan_mask.mean()),
        "n_rows_with_any_nan_sample": int(nan_mask.any(axis=1).sum()),
        "npz": npz_path,
        "flow_ckpt": str(cfg.restore_path),
        "flow_steps": int(cfg.agent.flow_steps),
        "n_rows": n, "k_samples": K_SAMPLES, "seed": int(cfg.seed),
        "n_invalid_in_sample": int((~valid).sum()),
        "per_row_mean_err": agg(err_v.mean(axis=1)),
        "per_row_min_err": agg(err_v.min(axis=1)),
        "per_row_max_err": agg(err_v.max(axis=1)),
        "point_roundtrip_reference": agg(err_pt[valid]),
        "mse_over_all_samples": float(np.mean(err_v ** 2)),
        "caveat": ("antmaze mixture arm is published as broken (ESS 7.6); every current "
                   "result uses the point arm there" if "antmaze" in npz_path else None),
    }
    write_report(report, cfg, "diag_preimage_sampling.json")


if __name__ == "__main__":
    main()
