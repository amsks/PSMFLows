"""D1: does the BC flow reproduce the dataset's action distribution?

If the flow does not cover the behaviour distribution, the latent family it induces
cannot index the behaviour policies either, and everything downstream inherits the gap.

Metrics (stdout JSON): RBF-MMD between dataset and flow actions at the SAME states;
k-means (numpy Lloyd's, k=16) mode histograms of dataset vs flow actions + their TV
distance; fraction of flow samples > eps from every dataset action at that state's
32 nearest-neighbor states (off-support fraction).

Run (real): JAX_PLATFORMS='' .venv/bin/python tools/validate_flow_fidelity.py \
    agent=fql env_name=cube-single-play-singletask-v0 \
    restore_path='/var/local/amsks/exp/<flow_run_dir>' restore_epoch=1000000
Smoke: add inversion.allow_untrained=true, preimage_limit=512.

On midi-01 add MUJOCO_GL=egl (no DISPLAY) and OGBENCH_DATASET_DIR=/var/local/amsks/ogbench.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see module docstring)

import hydra
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from omegaconf import OmegaConf

from agents.fql import FQLAgent
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.datasets import Dataset
from utils.flax_utils import restore_agent
from utils.log_utils import write_report


def _kmeans(x, k, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), k, replace=False)].copy()
    lab = np.zeros(len(x), np.int64)
    for _ in range(iters):
        d = ((x[:, None] - centers[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            if np.any(lab == j):
                centers[j] = x[lab == j].mean(0)
    return lab, centers


def _mmd_rbf(x, y, bw):
    def k(a, b):
        return np.exp(-((a[:, None] - b[None]) ** 2).sum(-1) / (2 * bw ** 2))
    return float(k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean())


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    _, _, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    n = min(int(cfg.get('preimage_limit', 4096) or 4096), ds.size)
    batch = ds.sample(n, idxs=np.random.default_rng(int(cfg.seed)).integers(0, ds.size, n))
    obs, act = batch['observations'], batch['actions']

    assert cfg.agent.agent_name == 'fql', 'run with agent=fql'
    agent_cfg = ml_collections.ConfigDict(_lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    agent = FQLAgent.create(cfg.seed, obs[:1], act[:1], agent_cfg)
    if cfg.restore_path is not None:
        agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)
    else:
        assert cfg.inversion.get('allow_untrained', False), (
            'a RANDOM flow reproduces nothing; set restore_path=<stage-A ckpt dir> '
            '(or inversion.allow_untrained=true for a plumbing smoke)')

    noises = jax.random.normal(jax.random.PRNGKey(cfg.seed), act.shape)
    flow_act = np.asarray(agent.compute_flow_actions(jnp.asarray(obs), noises=noises))

    # MMD cost is quadratic (an (m, m, d_a) intermediate), so cap the sample — but honor
    # a preimage_limit BELOW the cap instead of silently slicing [:1024] regardless, and
    # record the count so the estimate's support is visible in the report.
    m_mmd = min(n, 1024)
    m_bw = min(n, 512)
    bw = np.median(np.linalg.norm(act[:m_bw, None] - act[None, :m_bw], axis=-1)) + 1e-6
    lab_d, centers = _kmeans(act, k=min(16, n))
    lab_f = ((flow_act[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
    k = centers.shape[0]
    hist_d = np.bincount(lab_d, minlength=k) / len(lab_d)
    hist_f = np.bincount(lab_f, minlength=k) / len(lab_f)

    # off-support: distance from each flow action to the nearest dataset action among the
    # 32 states closest to its own state. The (n, n, ob) pairwise matrix bounds n at
    # ~4096; preimage_limit is its size knob.
    sd = ((obs[:, None] - obs[None]) ** 2).sum(-1)  # (n, n)
    nn = sd.argsort(1)[:, :32]
    off = np.array([np.min(np.linalg.norm(act[nn[i]] - flow_act[i], axis=-1)) for i in range(n)])

    report = {
        "flow": "TRAINED" if cfg.restore_path is not None else "RANDOM (control)",
        "env": cfg.env_name,
        "seed": int(cfg.seed),
        "n": int(n),
        "mmd_n": int(m_mmd),
        "mmd_rbf": _mmd_rbf(act[:m_mmd], flow_act[:m_mmd], bw),
        "mode_hist_tv": float(0.5 * np.abs(hist_d - hist_f).sum()),
        "off_support_frac@0.2": float((off > 0.2).mean()),
    }
    print(json.dumps(report, indent=2))
    write_report(report, cfg, "d1_flow_fidelity.json")


if __name__ == "__main__":
    main()
