"""Why-viz: the value landscape over latent space, with the data's preimages on top.

For a TRAINED psmflow checkpoint (08-05 latent-space-PSM semantics): at a few dataset
states, evaluates Q(u) = [mean_P - pess*unc](psi(s, w, u)^T w) on a dense u-grid
(oracle w from the dataset's true rewards, as D4), and overlays (a) the point preimages
of dataset actions taken at the nearest states — where the behavior data actually lives
in u-space — and (b) the latent gpi_select picks and (c) the actor's latent. One figure
answers: is the value landscape smooth in u, does selection land inside the data's
latent support, and does the actor agree with the argmax?

Grid is over u dims (0, 1) with remaining dims 0 — exact for pointmaze (d_a = 2), a
central slice elsewhere.

Run: JAX_PLATFORMS='' .venv/bin/python tools/viz_latent_value.py \
    agent=psmflow agent.preimage_path=<npz> agent.flow_ckpt_path=<flow_dir> \
    agent.flow_ckpt_epoch=500000 env_name=pointmaze-medium-navigate-singletask-task1-v0 \
    restore_path=<psmflow_run_dir> restore_epoch=500000
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

from agents import agents
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.datasets import Dataset
from utils.flax_utils import restore_agent
from utils.flow_inversion import load_augmented_dataset
from utils.log_utils import write_report

NUM_STATES = 4
GRID = 41           # GRID x GRID u-lattice per state
NEIGHBORS = 256     # dataset preimages overlaid = those of the NEIGHBORS nearest states


def _grid_q(agent, obs, u_grid, rng):
    """Q(u) = [mean_P - pess*unc](psi(s, w, u)^T w), gpi_select's statistic on a grid."""
    c = agent.config
    n = u_grid.shape[0]
    obs_b = jnp.broadcast_to(obs, (n, *obs.shape))
    w_b = jnp.broadcast_to(agent.task_z, (n, *agent.task_z.shape))
    from agents.psm import targets_uncertainty
    qpsis = agent.psi(obs_b, w_b, u_grid)
    Qs = (qpsis * agent.task_z).sum(-1)
    qmean, qunc = targets_uncertainty(Qs, c["num_parallel"])
    return np.asarray(qmean - c["actor_pessimism_penalty"] * qunc)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    env, eval_env, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    config = ml_collections.ConfigDict(_lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    assert config["agent_name"] == "psmflow"
    ex = ds.sample(1)
    agent = agents["psmflow"].create(cfg.seed, ex["observations"], ex["actions"], config)
    assert cfg.restore_path is not None, "needs a TRAINED psmflow checkpoint (restore_path)"
    agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    # Oracle w, seeded — same protocol as D4.
    n = min(ds.size, int(cfg.get("eval_relabel_size", 10000)))
    idxs = np.random.default_rng(int(cfg.seed)).integers(0, ds.size, n)
    zb = ds.sample(n, idxs=idxs)
    agent = agent.infer_eval_z(zb["next_observations"],
                               zb["rewards"] + float(cfg.get("eval_reward_shift", 1.0)))

    aug = load_augmented_dataset(config["preimage_path"])
    pts = np.asarray(aug["noise_preimage_point"])
    all_obs = np.asarray(aug["observations"])

    rng = np.random.default_rng(int(cfg.seed))
    state_idx = rng.integers(0, ds.size, NUM_STATES)
    lim = float(config["u_clip"])
    ax0 = np.linspace(-lim, lim, GRID)
    uu, vv = np.meshgrid(ax0, ax0)
    # action_dim is null in the yaml; create() fills it into the AGENT's config.
    d_a = int(agent.config["action_dim"])
    u_grid = np.zeros((GRID * GRID, d_a), np.float32)
    u_grid[:, 0], u_grid[:, 1] = uu.ravel(), vv.ravel()

    panels, report_states = [], []
    for si, idx in enumerate(state_idx):
        obs = jnp.asarray(all_obs[idx])
        Q = _grid_q(agent, obs, jnp.asarray(u_grid),
                    jax.random.PRNGKey(int(cfg.seed) + si)).reshape(GRID, GRID)
        u_star = np.asarray(agent.gpi_select(obs, seed=jax.random.PRNGKey(100 + si)))
        u_act = np.asarray(agent.config["u_clip"] * agent.actor(
            obs[None], agent.task_z[None], jnp.zeros((1, d_a))))[0]
        near = np.argsort(((all_obs - all_obs[idx]) ** 2).sum(-1))[:NEIGHBORS]
        u_near = pts[near]
        u_near = u_near[np.isfinite(u_near).all(-1)]
        panels.append((Q, u_near, u_star, u_act))
        report_states.append({
            "row": int(idx),
            "q_min": round(float(Q.min()), 4), "q_max": round(float(Q.max()), 4),
            "u_star_dims01": [round(float(u_star[0]), 3), round(float(u_star[1]), 3)],
            "u_actor_dims01": [round(float(u_act[0]), 3), round(float(u_act[1]), 3)],
            "q_at_u_star_grid": round(float(Q[np.abs(ax0 - u_star[1]).argmin(),
                                              np.abs(ax0 - u_star[0]).argmin()]), 4),
            "frac_neighbor_preimages_in_grid": round(
                float((np.abs(u_near[:, :2]) <= lim).all(-1).mean()), 3) if len(u_near) else None,
        })

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, ALERT = "#1f2430", "#6b7280", "#b3563d"
    fig, axes = plt.subplots(1, NUM_STATES, figsize=(3.6 * NUM_STATES, 3.4))
    for ax, (Q, u_near, u_star, u_act), meta in zip(np.atleast_1d(axes), panels, report_states):
        im = ax.imshow(Q, origin="lower", extent=(-lim, lim, -lim, lim),
                       cmap="Blues", aspect="equal")
        if len(u_near):
            ax.scatter(u_near[:, 0], u_near[:, 1], s=4, c=INK, alpha=0.35, lw=0,
                       label="data preimages (near states)")
        ax.scatter([u_star[0]], [u_star[1]], marker="*", s=140, c=ALERT,
                   edgecolors="white", lw=0.8, label="gpi_select")
        ax.scatter([u_act[0]], [u_act[1]], marker="P", s=110, c="#3a7d44",
                   edgecolors="white", lw=0.8, label="actor")
        ax.set_title(f"state row {meta['row']}", fontsize=8.5, color=INK)
        ax.tick_params(labelsize=7, colors=MUTED)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).ax.tick_params(labelsize=6)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(f"Q(u) = psi(s,w,u)^T w   {cfg.env_name}   (u dims 0,1"
                 + ("" if d_a == 2 else f" slice of {d_a}") + ")", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig_path = os.path.join(os.getcwd(), "viz_latent_value.png")
    fig.savefig(fig_path, dpi=150)

    report = {"env": cfg.env_name, "seed": int(cfg.seed), "grid": GRID,
              "restore_path": str(cfg.restore_path), "states": report_states,
              "figure": fig_path}
    print(json.dumps(report, indent=2))
    write_report(report, cfg, "viz_latent_value.json")


if __name__ == "__main__":
    main()
