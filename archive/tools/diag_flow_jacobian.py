"""Is there a preimage SET, or only a preimage point? Measure the flow map's Jacobian.

The method's second contribution assumes many latents decode to (nearly) the same action.
That is a statement about dG/du at the data: a direction the decoder squashes is a
direction you can move `u` along for free, so a genuine set exists; a well-conditioned
Jacobian means the map is locally injective and the preimage is a point, and any width you
add is label noise.

This reports the singular spectrum of dG(s, .)/du at each transition's own preimage — the
point the inversion actually solved for — so the numbers describe the region the mixture is
fitted in, not a random part of latent space.

Per environment (stdout JSON, one object; `report_out` also writes it):
  sigma_min / sigma_med / sigma_max   pooled over rows, and their per-row medians
  cond                                sigma_max / sigma_min per row
  flat_dirs_<eps>                     mean count of singular values below eps
  free_radius                         how far `u` can move along the smallest direction
                                      before the action moves by `action_tol` (= tol/sigma_min)

Run (real):
  MUJOCO_GL=egl OGBENCH_DATASET_DIR=/var/local/amsks/ogbench \\
  .venv/bin/python tools/diag_flow_jacobian.py agent=fql \\
      env_name=cube-single-play-singletask-v0 \\
      restore_path='/var/local/amsks/exp/PSMFLows/bcflow_cube_single_*/sd000_*' \\
      restore_epoch=500000 agent.flow_steps=100 preimage_limit=2048 \\
      report_out=/data-local/amsks/PSMFLows/logs/jacobian_cube.json

`agent.flow_steps` sets the inversion discretization and must be >= 100 for the same
reason the precompute asserts it: the implicit-Euler inverse diverges at 10.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax

import hydra
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from omegaconf import OmegaConf

from agents.fql import FQLAgent
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset
from utils.flax_utils import restore_agent

# Below this, a singular value is "flat": moving u along that direction by 1.0 moves the
# decoded action by less than eps. 0.1 is ~11% of the mean cube action norm (0.875).
FLAT_EPS = (0.01, 0.05, 0.1, 0.2)
ACTION_TOL = 0.05      # "the same action" budget for free_radius, in action units


def _lists_to_tuples(o):
    if isinstance(o, dict):
        return {k: _lists_to_tuples(v) for k, v in o.items()}
    if isinstance(o, list):
        return tuple(_lists_to_tuples(v) for v in o)
    return o


def _q(x, name):
    x = np.asarray(x, np.float64)
    return {f"{name}_mean": float(x.mean()), f"{name}_p05": float(np.percentile(x, 5)),
            f"{name}_med": float(np.median(x)), f"{name}_p95": float(np.percentile(x, 95))}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    _, _, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    n = min(int(cfg.get("preimage_limit", 2048) or 2048), ds.size)
    idxs = np.random.default_rng(int(cfg.seed)).integers(0, ds.size, n)
    batch = ds.sample(n, idxs=idxs)
    obs, act = np.asarray(batch["observations"]), np.asarray(batch["actions"])

    assert cfg.agent.agent_name == "fql", "run with agent=fql"
    n_steps = int(cfg.agent.flow_steps)
    assert n_steps >= 100, (
        f"agent.flow_steps={n_steps}: the implicit-Euler inverse diverges below 100 "
        "(diagnostic D3), so the Jacobian would be taken at a garbage point")
    agent_cfg = ml_collections.ConfigDict(_lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    agent = FQLAgent.create(cfg.seed, obs[:1], act[:1], agent_cfg)
    assert cfg.restore_path is not None, "a random flow has no meaningful Jacobian"
    agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    # jit(vmap): the un-jitted vmap of this function returns all-NaN on GPU (see
    # utils/flow_inversion.augment_dataset_with_point_preimage).
    jac_fn = jax.jit(jax.vmap(lambda s, a: agent._get_preimage_and_jacobian(s, a, n_steps)))
    d_a = act.shape[-1]
    us, jacs = [], []
    for start in range(0, n, 256):
        u, J = jac_fn(jnp.asarray(obs[start:start + 256]), jnp.asarray(act[start:start + 256]))
        us.append(np.asarray(u)); jacs.append(np.asarray(J))
    u = np.concatenate(us); J = np.concatenate(jacs)

    # The inverse diverges on a few rows; those Jacobians are meaningless, not flat.
    ok = np.isfinite(J).reshape(len(J), -1).all(1) & np.isfinite(u).all(1) & ((u ** 2).sum(-1) <= 100.0)
    J = J[ok]
    sv = np.linalg.svd(J.astype(np.float64), compute_uv=False)      # (n_ok, d_a), descending

    out = {
        "env_name": cfg.env_name,
        "restore_path": str(cfg.restore_path),
        "restore_epoch": int(cfg.restore_epoch or 0),
        "flow_steps": n_steps,
        "action_dim": int(d_a),
        "rows": int(len(sv)),
        "rows_dropped": int((~ok).sum()),
        "action_norm_mean": float(np.linalg.norm(np.clip(act, -1, 1), axis=-1).mean()),
        # pooled over every (row, direction) pair — comparable across action dims
        "sigma_all": _q(sv.reshape(-1), "sigma"),
        "sigma_min": _q(sv[:, -1], "sigma_min"),
        "sigma_max": _q(sv[:, 0], "sigma_max"),
        "cond": _q(sv[:, 0] / np.maximum(sv[:, -1], 1e-12), "cond"),
        # how far u travels along the flattest direction for an ACTION_TOL action change
        "free_radius": _q(ACTION_TOL / np.maximum(sv[:, -1], 1e-12), "free_radius"),
        "action_tol": ACTION_TOL,
        "flat_dirs": {str(e): float((sv < e).sum(1).mean()) for e in FLAT_EPS},
        "flat_dirs_frac": {str(e): float((sv < e).mean()) for e in FLAT_EPS},
        # per-direction medians: is the flatness one direction or spread over many?
        "sigma_by_rank_med": [float(np.median(sv[:, k])) for k in range(d_a)],
    }
    print("RESULT " + json.dumps(out))
    if cfg.get("report_out", None):
        with open(cfg.report_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"report -> {cfg.report_out}")


if __name__ == "__main__":
    main()
