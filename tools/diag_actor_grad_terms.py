"""Per-term gradient norms of the latent actor's loss, on a restored checkpoint.

`flow_actor_loss` sums three terms; this reloads a checkpoint, draws one batch, and
reports the global gradient norm each term contributes to the actor (and to the CFM
velocity field, which only `bc_flow_loss` touches).

It also measures what the actor climbs under `policy_index=latent`:

  q_single    -Q/|Q| at the ONE prior index draw u' the batch sampled, i.e. the
              `flow_actor_loss` computation at actor.index_panel=0;
  q_panel     the same with Q replaced by the max over a panel of `index_panel` prior
              draws u' -- the object `gpi_select` maximises.

A low cosine between the two gradients means the actor is optimising a different
objective from the deployed GPI rule.

Run (CPU is fine, one batch):
  .venv/bin/python tools/diag_actor_grad_terms.py agent=psmflow \
      agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \
      agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz \
      env_name=cube-single-play-singletask-v0 \
      restore_path=<run_dir> restore_epoch=500000
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see agents/psmflow.py invariants)

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


def _gnorm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves)))


def _flat(tree):
    return jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(tree)])


def _cos(a, b):
    fa, fb = _flat(a), _flat(b)
    return float(fa @ fb / (jnp.linalg.norm(fa) * jnp.linalg.norm(fb) + 1e-12))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    np.random.seed(int(cfg.seed))
    _, _, _train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    cli_agent = OmegaConf.to_container(cfg.agent, resolve=True)
    merged, prov = merge_run_config(cli_agent, cfg.restore_path, _cli_agent_keys())
    config = ml_collections.ConfigDict(_lists_to_tuples(merged))

    aug = load_augmented_dataset(config["preimage_path"])
    aug, _ = repair_invalid_preimages(aug)
    ds = Dataset.create(**aug)
    ds.return_preimage_noise = True
    ds.preimage_point_mode = bool(config.get("use_point_preimage", False))

    ex = ds.sample(1)
    agent = agents[config["agent_name"]].create(cfg.seed, ex["observations"], ex["actions"], config)
    assert cfg.restore_path is not None, "needs a trained checkpoint (restore_path)"
    agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    batch = ds.sample(int(config["batch_size"]))
    rng = jax.random.PRNGKey(int(cfg.seed))
    sampled = agent.sample_step_inputs(batch, rng)

    c = agent.config
    obs = batch["observations"]
    w = sampled.task_w
    x0, t, noise = sampled.flow_x0, sampled.flow_t, sampled.flow_noise
    u_clip, steps = c["u_clip"], c["actor"]["flow_steps"]
    panel, d_a = int(c["index_panel"]), int(c["action_dim"])

    def rollout(vf_p, o, n):
        u = n
        for i in range(steps):
            ti = jnp.full((o.shape[0], 1), i / steps)
            u = u + agent.actor_vf(o, u, ti, params=vf_p) / steps
        return jnp.clip(u, -u_clip, u_clip)

    tgt = jax.lax.stop_gradient(rollout(agent.actor_vf.params, obs, noise))

    def q_readout(u_a, index):
        Qs = (agent.psi(obs, index, u_a) * w).sum(-1)
        qmean, qunc = targets_uncertainty(Qs, c["num_parallel"])
        return qmean - c["actor_pessimism_penalty"] * qunc, Qs

    def q_single(actor_params):
        u_a = u_clip * agent.actor(obs, w, noise, params=actor_params)
        Q, Qs = q_readout(u_a, agent._index(sampled))
        return -Q.mean() / jax.lax.stop_gradient(jnp.abs(Qs).mean() + 1e-8)

    def q_panel(actor_params):
        """Max over an index panel: the object gpi_select maximises."""
        u_a = u_clip * agent.actor(obs, w, noise, params=actor_params)
        key = jax.random.fold_in(rng, 991)
        u_idx = jnp.clip(jax.random.normal(key, (panel, obs.shape[0], d_a)), -u_clip, u_clip)
        obs_r = jnp.broadcast_to(obs[None], (panel, *obs.shape)).reshape(panel * obs.shape[0], -1)
        w_r = jnp.broadcast_to(w[None], (panel, *w.shape)).reshape(panel * obs.shape[0], -1)
        u_r = jnp.broadcast_to(u_a[None], (panel, *u_a.shape)).reshape(panel * obs.shape[0], d_a)
        Qs = (agent.psi(obs_r, u_idx.reshape(panel * obs.shape[0], d_a), u_r) * w_r).sum(-1)
        qmean, qunc = targets_uncertainty(Qs, c["num_parallel"])
        Q = (qmean - c["actor_pessimism_penalty"] * qunc).reshape(panel, obs.shape[0])
        return -Q.max(0).mean() / jax.lax.stop_gradient(jnp.abs(Qs).mean() + 1e-8)

    def distill(actor_params):
        u_a = u_clip * agent.actor(obs, w, noise, params=actor_params)
        return jnp.mean((u_a - tgt) ** 2)

    def bc_flow(vf_params):
        xt = (1 - t) * x0 + t * sampled.u_data
        return jnp.mean((agent.actor_vf(obs, xt, t, params=vf_params) - (sampled.u_data - x0)) ** 2)

    ap = agent.actor.params
    g_q, v_q = jax.value_and_grad(q_single)(ap)[1], q_single(ap)
    g_qp, v_qp = jax.value_and_grad(q_panel)(ap)[1], q_panel(ap)
    g_d, v_d = jax.value_and_grad(distill)(ap)[1], distill(ap)
    g_f, v_f = jax.value_and_grad(bc_flow)(agent.actor_vf.params)[1], bc_flow(agent.actor_vf.params)
    bc_coeff = float(c["actor"]["bc_coeff"])
    g_d_w = jax.tree_util.tree_map(lambda x: bc_coeff * x, g_d)

    # What the actor's latent is worth against the deployed GPI selection, same batch.

    u_a = u_clip * agent.actor(obs, w, noise)
    key = jax.random.fold_in(rng, 992)
    K = int(c["gpi_num_u"])
    n = min(64, obs.shape[0])
    o_n, w_n = obs[:n], w[:n]
    u_cand = jnp.clip(jax.random.normal(key, (K, n, d_a)), -u_clip, u_clip)
    u_idx = jnp.clip(jax.random.normal(jax.random.fold_in(rng, 993), (K, n, d_a)), -u_clip, u_clip)

    def score(u_k, i_k):
        Qs = (agent.psi(o_n, i_k, u_k) * w_n).sum(-1)
        qm, qu = targets_uncertainty(Qs, c["num_parallel"])
        return qm - c["actor_pessimism_penalty"] * qu

    pair = jax.vmap(lambda ik: jax.vmap(lambda uk: score(uk, ik))(u_cand))(u_idx)   # (K_idx, K_u, n)
    gpi_best = pair.max(0).max(0)                       # per-state GPI value
    actor_val = jax.vmap(lambda ik: score(u_a[:n], ik))(u_idx).max(0)
    scale = jnp.abs(pair).mean() + 1e-8

    report = {
        "restore_path": str(cfg.restore_path), "restore_epoch": int(cfg.restore_epoch),
        "env": cfg.env_name, "agent_config_source": prov,
        "psi_form": config.get("psi_form"), "policy_index": config.get("policy_index"),
        "index_agg": config.get("index_agg"), "train_actor": config.get("train_actor"),
        "acting": config.get("acting"), "bc_coeff": bc_coeff,
        "batch_size": int(config["batch_size"]), "index_panel": panel, "gpi_num_u": K,
        "loss_values": {"q_single": float(v_q), "q_panel_max": float(v_qp),
                        "distill": float(v_d), "distill_weighted": bc_coeff * float(v_d),
                        "bc_flow": float(v_f)},
        "actor_grad_norms": {"q_single": _gnorm(g_q), "q_panel_max": _gnorm(g_qp),
                             "distill_weighted": _gnorm(g_d_w), "distill_raw": _gnorm(g_d)},
        "actor_vf_grad_norm_bc_flow": _gnorm(g_f),
        "ratio_distill_over_q_single": _gnorm(g_d_w) / (_gnorm(g_q) + 1e-12),
        "ratio_distill_over_q_panel": _gnorm(g_d_w) / (_gnorm(g_qp) + 1e-12),
        "cos_q_single_vs_q_panel": _cos(g_q, g_qp),
        "cos_q_single_vs_distill": _cos(g_q, g_d),
        "gpi_vs_actor_value": {
            "gpi_best_mean": float(gpi_best.mean()),
            "actor_value_mean": float(actor_val.mean()),
            "gap_rel": float((gpi_best - actor_val).mean() / scale),
            "q_spread_over_u_rel": float(pair.max(1).std(0).mean() / scale),
        },
    }
    print(OmegaConf.to_yaml(OmegaConf.create(
        {k: v for k, v in report.items() if k != "agent_config_source"})))
    write_report(report, cfg, "diag_actor_grad_terms.json")


if __name__ == "__main__":
    main()
