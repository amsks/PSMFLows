"""Where does an amortized latent actor's chosen u sit inside the GPI roster?

The 2026-09-07 puzzle (docs/design/2026-09-06-dsrl-actor-audit.md §4g): the DSRL-NA actor's
regression explains only ~6% of the GPI argmax by MSE, `prior_shrunk` rules out the support
explanation, and yet the arm scores 0.307 against the actor-free arm's 0.415 on cube. MSE is
a harsh metric -- it is dominated by wrong-MODE cases that a ranking ignores -- so the actor
could still be a partial distillation that the regression number under-reads.

This scores, per state, a FIXED roster of candidate latents and asks where the actor's own
latent falls in it:

  (a) rank_percentile   the actor's GPI score as a percentile of the roster's scores
                        (0.5 = chance, 1.0 = the actor picks the roster argmax);
  (b) spearman_actor_vs_rostermax   across states, does the actor score high exactly where
                        the roster's best is high;
  (c) dist_to_argmax vs dist_to_roster_mean   is the actor's latent NEAR the argmax in
                        latent space, or merely somewhere typical.

`--` all against the same GPI score `gpi_select` maximises: per candidate u, the max over an
index panel of [mean_P - pessimism * unc] psi(s, u, u')^T w, with w the EVAL task vector from
`infer_eval_z` (what acting actually uses), not a training draw.

Caveat that must travel with any reading of (a): the 2026-09-05 selection diagnostic
measured the critic's own per-u ranking as barely reproducible (Spearman 0.21-0.36 between
checkpoints, and the argmax scoring WORSE than random on an MC ground truth). A low
percentile here therefore does not by itself mean the actor is bad -- it can equally mean
the GPI score is a poor ranking of action latents, which is already on record.

Run:
  .venv/bin/python tools/diag_actor_vs_gpi.py agent=psmflow \\
      agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \\
      agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz \\
      env_name=cube-single-play-singletask-v0 \\
      restore_path=<run_dir> restore_epoch=300000 report_out=<json>
"""
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

from agents import agents
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from tools.eval_checkpoint import _cli_agent_keys, merge_run_config
from utils.datasets import Dataset
from utils.flax_utils import restore_agent
from utils.flow_inversion import load_augmented_dataset, repair_invalid_preimages
from utils.log_utils import write_report
from utils.psm_common import targets_uncertainty

N_STATES = 256
N_CAND = 64      # roster of action latents u, shared across states
N_INDEX = 64     # policy indices u' the GPI score maxes over (gpi_num_u's value)


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / d) if d > 0 else 0.0


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    np.random.seed(int(cfg.seed))
    _, _, _td, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    merged, prov = merge_run_config(OmegaConf.to_container(cfg.agent, resolve=True),
                                    cfg.restore_path, _cli_agent_keys())
    config = ml_collections.ConfigDict(_lists_to_tuples(merged))

    aug, _ = repair_invalid_preimages(load_augmented_dataset(config["preimage_path"]))
    ds = Dataset.create(**aug)
    ds.return_preimage_noise = True
    ds.preimage_point_mode = bool(config.get("use_point_preimage", False))

    ex = ds.sample(1)
    agent = agents[config["agent_name"]].create(cfg.seed, ex["observations"], ex["actions"], config)
    agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    # Eval task vector, exactly as tools/eval_checkpoint.py builds it.
    zb = ds.sample(min(ds.size, int(cfg.get("eval_relabel_size", 10000))))
    agent = agent.infer_eval_z(zb["next_observations"],
                              zb["rewards"] + float(cfg.get("eval_reward_shift", 1.0)))

    c = agent.config
    d_a, u_clip = int(c["action_dim"]), float(c["u_clip"])
    batch = ds.sample(N_STATES)
    obs = batch["observations"]
    w = jnp.broadcast_to(agent.task_z, (N_STATES, agent.task_z.shape[0]))

    # ONE roster, shared across states, so every state is scored against the same candidates.
    rk = jax.random.PRNGKey(20260907)
    k_u, k_i, k_n = jax.random.split(rk, 3)
    roster = jnp.clip(jax.random.normal(k_u, (N_CAND, d_a)), -u_clip, u_clip)
    u_idx = jnp.clip(jax.random.normal(k_i, (N_INDEX, d_a)), -u_clip, u_clip)
    idx_b = jnp.broadcast_to(u_idx[:, None, :], (N_INDEX, N_STATES, d_a))

    def gpi_score(u_b):
        """[mean_P - pess*unc] psi(s, u, u')^T w, maxed over the index panel. u_b: (B, d_a)."""
        Qk = agent._psi_q_over_indices(obs, u_b, w, idx_b)          # (P, N_INDEX, B)
        qm, qu = targets_uncertainty(Qk, c["num_parallel"])
        return (qm - c["actor_pessimism_penalty"] * qu).max(0)      # (B,)

    scores = np.stack([np.asarray(gpi_score(jnp.broadcast_to(roster[m], (N_STATES, d_a))))
                       for m in range(N_CAND)])                     # (N_CAND, B)

    # The actor's own latent, through whichever head this checkpoint carries.
    noise = jax.random.normal(k_n, (N_STATES, d_a))
    if c["actor_mode"] == "ddpg":
        u_a = u_clip * agent.actor(obs, w, noise)
    else:
        u_a = agent._sac_latent(obs, w, noise)[0]
    actor_score = np.asarray(gpi_score(u_a))                        # (B,)
    u_a = np.asarray(u_a)

    # (a) percentile of the actor's score within the roster, per state.
    pct = (scores < actor_score[None]).mean(0)                      # (B,)
    roster_max = scores.max(0)
    roster_argmax = np.asarray(roster)[scores.argmax(0)]            # (B, d_a)

    # (c) latent-space distances.
    d_arg = np.linalg.norm(u_a - roster_argmax, axis=-1)
    d_all = np.linalg.norm(u_a[None] - np.asarray(roster)[:, None], axis=-1)  # (N_CAND, B)
    d_mean = d_all.mean(0)

    report = {
        "restore_path": str(cfg.restore_path), "restore_epoch": int(cfg.restore_epoch),
        "env": cfg.env_name, "train_seed": prov.get("train_seed"),
        "actor_mode": c["actor_mode"], "psi_form": c["psi_form"],
        "acting": c["acting"], "train_actor": bool(c["train_actor"]),
        "actor_index_panel": int(c["actor"]["index_panel"]),
        "n_states": N_STATES, "n_candidates": N_CAND, "n_index": N_INDEX,
        # (a)
        "rank_percentile_mean": float(pct.mean()),
        "rank_percentile_median": float(np.median(pct)),
        "frac_states_top_quartile": float((pct >= 0.75).mean()),
        "frac_states_top_decile": float((pct >= 0.90).mean()),
        "frac_states_below_chance": float((pct < 0.5).mean()),
        # (b)
        "spearman_actor_vs_rostermax": _spearman(actor_score, roster_max),
        "spearman_actor_vs_rostermean": _spearman(actor_score, scores.mean(0)),
        # (c)
        "dist_to_roster_argmax": float(d_arg.mean()),
        "dist_to_roster_mean_candidate": float(d_mean.mean()),
        "dist_ratio_argmax_over_typical": float(d_arg.mean() / (d_mean.mean() + 1e-12)),
        "actor_u_norm": float(np.linalg.norm(u_a, axis=-1).mean()),
        "roster_u_norm": float(np.linalg.norm(np.asarray(roster), axis=-1).mean()),
        # scale context for the scores themselves
        "actor_score_mean": float(actor_score.mean()),
        "roster_max_mean": float(roster_max.mean()),
        "roster_mean_mean": float(scores.mean()),
        "score_gap_rel": float((roster_max - actor_score).mean()
                               / (np.abs(scores).mean() + 1e-12)),
        "agent_config_source": prov,
    }
    for k, v in report.items():
        if k != "agent_config_source":
            print(f"{k}: {v}")
    write_report(report, cfg, "diag_actor_vs_gpi.json")


if __name__ == "__main__":
    main()
