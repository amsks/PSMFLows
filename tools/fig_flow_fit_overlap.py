"""The one intuitive picture: does the flow's action cloud sit on top of the data's?

Every other flow-fit figure is a diagnostic answering a narrow question (memorization,
sample-budget sensitivity, recall vs radius). None of them lets you simply LOOK at the fit.
This one does: pick a handful of states, draw the frozen flow's decoded actions and the
actions actually recorded at nearby states, project both into the same 2-D plane, and see
whether they overlap.

Per panel (one anchor state):
  grey cloud   N decoded actions, u ~ N(0, I) unclipped, exact ODE at `flow_decode_steps`
  orange x     the recorded actions of the k nearest dataset states (the local behaviour)
  blue star    THIS state's own recorded action
  red +        the decode of the stored point preimage, when a preimage npz is given --
               the exact inverse, i.e. what the flow can hit given the right latent

Projection is a 2-D PCA fitted per panel on the union of both clouds, so the view is the
one that best separates what is actually there rather than an arbitrary pair of action
dimensions. It IS a projection: overlap in 2-D is necessary, not sufficient. The quantitative
statements live in `tools/diag_flow_fit.py`; this figure is for reading at a glance.

Run:
  MUJOCO_GL=egl .venv/bin/python tools/fig_flow_fit_overlap.py agent=psmflow \
      env_name=cube-single-play-singletask-v0 \
      agent.flow_ckpt_path=$FLOW agent.flow_ckpt_epoch=500000 \
      agent.gpi_decode=ode agent.flow_decode_steps=100 \
      +fit_preimage=$PSM_DATA/preimages/cube-single-play.npz \
      +fig_out=$PSM_DATA/figs/flow_fit_overlap_cube.png
"""
# ruff: noqa: I001 -- import ORDER is load-bearing. `sys.path` first so `utils` resolves,
# then xla_guard, then jax. Do not let `ruff --fix` sort this block: it moves xla_guard
# below jax and silently breaks the ODE.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see module docstring)

import hydra
import jax
import jax.numpy as jnp
import matplotlib
import ml_collections
import numpy as np
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agents import agents  # noqa: E402
from envs.env_utils import make_env_and_datasets  # noqa: E402
from main import _lists_to_tuples  # noqa: E402
from utils.datasets import Dataset  # noqa: E402
from utils.geometry import NeighbourIndex  # noqa: E402

N_PANELS = 9
N_SAMPLES = 256
K_NEIGH = 32
SEED = 0


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    _, _, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    config = ml_collections.ConfigDict(
        _lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    ex = ds.sample(1)
    agent = agents["psmflow"].create(cfg.seed, ex["observations"], ex["actions"], config)
    d_a = int(agent.config["action_dim"])

    obs_all = np.asarray(ds["observations"])
    act_all = np.asarray(ds["actions"])
    index = NeighbourIndex(obs_all, act_all, seed=SEED)

    rng = np.random.default_rng(SEED)
    anchors = rng.choice(len(obs_all), N_PANELS, replace=False)

    # Optional: the exact inverse, as a reference point inside each panel.
    point_pre = None
    pre = cfg.get("fit_preimage", None)
    if pre and os.path.exists(str(pre)):
        with np.load(str(pre)) as z:
            point_pre = z["noise_preimage_point"][anchors]

    fig, axes = plt.subplots(3, 3, figsize=(11.5, 11.0))
    key = jax.random.PRNGKey(SEED)
    for i, (ax, row) in enumerate(zip(axes.ravel(), anchors)):
        s = obs_all[row]
        key, k = jax.random.split(key)
        u = jax.random.normal(k, (N_SAMPLES, d_a))
        obs_b = jnp.broadcast_to(jnp.asarray(s), (N_SAMPLES, *s.shape))
        flow = np.asarray(agent.decode(obs_b, u))

        _, pos = index.query(s[None], k=K_NEIGH, exclude_rows=np.asarray([row]))
        ball = index.act[pos[0]]
        own = act_all[row]

        stack = np.concatenate([flow, ball, own[None]], 0)
        mu = stack.mean(0)
        # PCA on the union: the 2-D view that best shows whatever structure is present.
        _, _, vt = np.linalg.svd(stack - mu, full_matrices=False)
        proj = lambda x: (np.atleast_2d(x) - mu) @ vt[:2].T  # noqa: E731

        f2, b2, o2 = proj(flow), proj(ball), proj(own)
        ax.scatter(f2[:, 0], f2[:, 1], s=14, c="#9a9a94", alpha=0.45, linewidths=0,
                   label=f"flow decodes (N={N_SAMPLES})" if i == 0 else None)
        ax.scatter(b2[:, 0], b2[:, 1], s=52, marker="x", c="#eb6834", linewidths=1.8,
                   label=f"recorded actions, {K_NEIGH} nearest states" if i == 0 else None)
        ax.scatter(o2[:, 0], o2[:, 1], s=210, marker="*", c="#2a78d6",
                   edgecolors="white", linewidths=0.9, zorder=5,
                   label="this state's own recorded action" if i == 0 else None)
        if point_pre is not None:
            pp = np.asarray(agent.decode(jnp.asarray(s)[None], jnp.asarray(point_pre[i])[None]))
            p2 = proj(pp)
            ax.scatter(p2[:, 0], p2[:, 1], s=150, marker="+", c="#d03b3b", linewidths=2.2,
                       zorder=6, label="decode of the exact preimage" if i == 0 else None)

        # How far the nearest of N decodes lands from this state's own action, normalized
        # the way every other number in the flow-fit family is.
        d = np.linalg.norm(flow - own[None], axis=1).min() / index.action_scale
        ax.set_title(f"state {row}   best-of-{N_SAMPLES} to own action: {d:.3f}", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#d8d7d2")

    env_short = str(cfg.env_name).replace("-singletask-v0", "")
    fig.suptitle(
        f"Does the behaviour flow's action cloud cover the data?  —  {env_short}\n"
        f"2-D PCA per panel, fitted on both clouds together; distances in units of mean|a| "
        f"= {index.action_scale:.3f}",
        fontsize=12.5)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0.045, 1, 0.945])

    out = str(cfg.get("fig_out", "flow_fit_overlap.png"))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.replace(".png", ".pdf"))
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
