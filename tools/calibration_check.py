"""Calibration: does psi^T w predict what fixed-u policies actually achieve?

The GPI failure signature (write-up Rem. 9.2) is a representation that over-scores some
latent the argmax then picks — an adversarial search against psi. This measures it
directly: at eval start states, score K candidate latents exactly as gpi_select does
(oracle w, as D4), then roll each candidate out as a FIXED-u policy and compute its
realized discounted return under the true task reward. Reports per-state Pearson and
Spearman correlation of predicted score vs realized return, plus the realized RANK of
the candidate gpi_select would pick (1.0 = the pick was genuinely the best candidate).

Scales of the two axes differ (w is normalized, env reward is not) — the calibration
claim is about RANKING, which is all the argmax uses.

Run: MUJOCO_GL=egl .venv/bin/python tools/calibration_check.py \
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
from utils.log_utils import write_report

NUM_STATES = 4      # eval-episode start states (env reset seeds)
NUM_CAND = 16       # candidate latents scored + rolled out per state
MAX_STEPS = 200


def _score(agent, obs, u_cand, rng):
    """Predicted value per candidate, exactly gpi_select's statistic."""
    from agents.psm import targets_uncertainty
    c = agent.config
    k, mi = u_cand.shape[0], c["gpi_num_uprime"]
    u_idx = jnp.clip(jax.random.normal(rng, (mi, c["action_dim"])), -c["u_clip"], c["u_clip"])
    obs_b = jnp.broadcast_to(obs, (k * mi, *obs.shape))
    uc, ui = jnp.repeat(u_cand, mi, axis=0), jnp.tile(u_idx, (k, 1))
    Qs = (agent.psi(obs_b, ui, uc) * agent.task_z).sum(-1)
    qmean, qunc = targets_uncertainty(Qs, c["num_parallel"])
    return np.asarray((qmean - c["actor_pessimism_penalty"] * qunc).reshape(k, mi).max(axis=1))


def _pearson(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 3)


def _spearman(a, b):
    return _pearson(np.argsort(np.argsort(a)).astype(float),
                    np.argsort(np.argsort(b)).astype(float))


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

    n = min(ds.size, int(cfg.get("eval_relabel_size", 10000)))
    idxs = np.random.default_rng(int(cfg.seed)).integers(0, ds.size, n)
    zb = ds.sample(n, idxs=idxs)
    shift = float(cfg.get("eval_reward_shift", 1.0))
    agent = agent.infer_eval_z(zb["next_observations"], zb["rewards"] + shift)

    gamma = float(config["discount"])
    onestep = jax.jit(agent.decode)
    rng = np.random.default_rng(int(cfg.seed))
    states, preds, reals = [], [], []
    for si in range(NUM_STATES):
        u_cand = np.clip(rng.standard_normal((NUM_CAND, int(config["action_dim"]))),
                         -float(config["u_clip"]), float(config["u_clip"])).astype(np.float32)
        ob0, _ = eval_env.reset(seed=int(cfg.seed) * 1000 + si)
        pred = _score(agent, jnp.asarray(ob0), jnp.asarray(u_cand),
                      jax.random.PRNGKey(int(cfg.seed) + si))
        real = np.zeros(NUM_CAND)
        for k in range(NUM_CAND):
            ob, _ = eval_env.reset(seed=int(cfg.seed) * 1000 + si)  # same start per candidate
            g, disc = 0.0, 1.0
            for _ in range(MAX_STEPS):
                a = np.asarray(onestep(jnp.asarray(ob)[None], jnp.asarray(u_cand[k])[None]))[0]
                ob, r, term, trunc, _ = eval_env.step(np.clip(a, -1, 1))
                g += disc * (float(r) + shift)
                disc *= gamma
                if term or trunc:
                    break
            real[k] = g
        pick = int(np.argmax(pred))
        states.append({
            "reset_seed": int(cfg.seed) * 1000 + si,
            "pearson": _pearson(pred, real),
            "spearman": _spearman(pred, real),
            # realized rank of the argmax pick, in (0, 1]; 1.0 = the pick was the true best
            "pick_realized_rank": round(float((real <= real[pick]).mean()), 3),
            "pred": [round(float(x), 4) for x in pred],
            "real": [round(float(x), 3) for x in real],
        })
        preds.append(pred), reals.append(real)

    spearmen = [s["spearman"] for s in states if s["spearman"] is not None]
    report = {
        "env": cfg.env_name, "seed": int(cfg.seed),
        "restore_path": str(cfg.restore_path), "restore_epoch": int(cfg.restore_epoch),
        "num_states": NUM_STATES, "num_candidates": NUM_CAND,
        "mean_spearman": round(float(np.mean(spearmen)), 3) if spearmen else None,
        "mean_pick_realized_rank": round(float(np.mean([s["pick_realized_rank"] for s in states])), 3),
        "states": states,
    }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, BLUE, ALERT = "#1f2430", "#6b7280", "#3d6fb0", "#b3563d"
    fig, axes = plt.subplots(1, NUM_STATES, figsize=(3.2 * NUM_STATES, 3.1))
    for ax, s, pred, real in zip(np.atleast_1d(axes), states, preds, reals):
        ax.scatter(pred, real, s=22, c=BLUE, lw=0)
        pick = int(np.argmax(pred))
        ax.scatter([pred[pick]], [real[pick]], marker="*", s=150, c=ALERT,
                   edgecolors="white", lw=0.8, label="gpi pick")
        ax.set_title(f"spearman {s['spearman']}  pick-rank {s['pick_realized_rank']}",
                     fontsize=8, color=INK)
        ax.set_xlabel("predicted psi^T w", fontsize=7.5, color=MUTED)
        ax.tick_params(labelsize=7, colors=MUTED)
    np.atleast_1d(axes)[0].set_ylabel("realized disc. return", fontsize=7.5, color=MUTED)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"calibration: predicted vs realized per fixed-u candidate   {cfg.env_name}",
                 fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig_path = os.path.join(os.getcwd(), "calibration_check.png")
    fig.savefig(fig_path, dpi=150)
    report["figure"] = fig_path

    print(json.dumps(report, indent=2))
    write_report(report, cfg, "calibration_check.json")


if __name__ == "__main__":
    main()
