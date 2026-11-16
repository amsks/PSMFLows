"""Read-only numeric check (2026-11-09 port audit): on one dataset batch, compare the actor objective of
density_fb_latent (L) and psmgoal with the Density FB recipe (P) at trained checkpoints.

Per checkpoint it reports
  * the index's own reward view vs the hindsight kernel reward on the ACTOR rows (proxy correlation),
  * the slow-feature moments (trace and participation ratio of the stored covariance),
  * the norm of d/du of the Q term (-q_coeff Q.mean()/q_scale) and of the BC term (bc_coeff ||u - rollout||^2)
    at the actor's own u, and their ratio,
  * how much Q moves with u at a fixed (s, index): std over K prior draws, divided by q_scale.
Run: JAX_PLATFORMS=cpu .venv/bin/python psmgoal-vs-dfb-2026-11-09/actor_grad_check.py <out.json>
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.xla_guard  # noqa: F401  (before jax)
import json
import glob

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset
from utils.flax_utils import restore_agent
from utils.flow_inversion import load_augmented_dataset, repair_invalid_preimages
from agents.psmgoal import kernel_reward, parse_hit_spec

EXP = "/mnt/home/amohan/psm-data/exp/PSMFLows"
FLOW = "/mnt/home/amohan/psm-data/flow/cube-single-play"
PRE = "/mnt/home/amohan/psm-data/preimages/cube-single-play.npz"
K_DRAWS, N_SENS = 32, 32


def tup(x):
    if isinstance(x, dict):
        return {k: tup(v) for k, v in x.items()}
    if isinstance(x, list):
        return tuple(tup(v) for v in x)
    return x


def load(run_dir, epoch, base):
    flags = json.load(open(os.path.join(run_dir, "flags.json")))
    a = flags["agent"]
    a["flow_ckpt_path"], a["preimage_path"] = FLOW, PRE
    cfg = ml_collections.ConfigDict(tup(a))
    ds = dict(base)
    ds = Dataset.create(**ds)
    ds.return_preimage_noise = True
    ds.preimage_point_mode = bool(cfg.get("use_point_preimage", False))
    ds.return_goals = True
    ds.return_fb_goals = True
    if cfg["agent_name"] == "density_fb_latent":
        from agents.density_fb_latent import dataset_goal_knobs
        for k, v in dataset_goal_knobs(cfg).items():
            setattr(ds, k, v)
    else:
        ds.return_index = True
        ds.goal_discount = float(cfg["goal_discount"])
        ds.goal_random_frac = float(cfg["goal_random_frac"])
        ds.goal_cur_frac = float(cfg["goal_cur_frac"])
        ds.goal_sampling = str(cfg["goal_sampling"])
    ex = ds.sample(1)
    agent = agents[cfg["agent_name"]].create(0, ex["observations"], ex["actions"], cfg)
    agent = restore_agent(agent, run_dir, epoch)
    return agent, ds, cfg


def corr_rows(a, b, mask):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    c = (a * b).sum(1) / jnp.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)
    m = mask.reshape(-1)
    return float((c * m).sum() / jnp.maximum(m.sum(), 1.0))


def moments(mu, cov):
    ev = np.linalg.eigvalsh(np.asarray(cov))[::-1]
    return {"mu_sq": float(np.asarray(mu) @ np.asarray(mu)), "cov_trace": float(ev.sum()),
            "cov_pr": float(ev.sum() ** 2 / (ev ** 2).sum()), "cov_top5": [float(x) for x in ev[:5]]}


def check_L(agent, batch, cfg):
    c = agent.config
    inj = agent._draw_injection(batch, jax.random.PRNGKey(1))
    obs = jnp.asarray(batch["observations"])
    z_a, x_in, basis = inj["z_a"], inj["x_in"], inj["zenc_u"]
    view = basis @ z_a.T / jnp.sqrt(float(c["z_dim"]))                         # (M, N)
    kern = kernel_reward(jnp.asarray(batch["fb_goals"]), jnp.asarray(batch["next_observations"]),
                         parse_hit_spec(c["fb_hit_spec"]))
    proxy = corr_rows(view.T, kern, inj["hindsight_a"])
    a_cfg = c["actor"]
    noise = inj["actor_noise"]
    u = agent.actor_u(obs, z_a, noise)
    tgt = agent._rollout(agent.params["actor_vf"], obs, noise)
    q, qs = agent.q_fn(agent.params, obs, z_a, u, x_in, basis)
    q_scale = jnp.maximum(jnp.abs(qs).mean(), 1e-6)

    def qterm(uu):
        return -float(a_cfg["q_coeff"]) * agent.q_fn(agent.params, obs, z_a, uu, x_in, basis)[0].mean() / q_scale

    def bcterm(uu):
        return float(a_cfg["bc_coeff"]) * jnp.mean((uu - tgt) ** 2)
    gq, gb = jax.grad(qterm)(u), jax.grad(bcterm)(u)
    # Q's u-sensitivity at fixed (s, z): K clipped prior draws on N_SENS rows
    n = N_SENS
    eps = jnp.clip(jax.random.normal(jax.random.PRNGKey(2), (n, K_DRAWS, u.shape[-1])), -3.0, 3.0)
    o = jnp.repeat(obs[:n], K_DRAWS, 0)
    zz = jnp.repeat(z_a[:n], K_DRAWS, 0)
    qd = agent.q_fn(agent.params, o, zz, eps.reshape(n * K_DRAWS, -1), x_in, basis)[0].reshape(n, K_DRAWS)
    return {"proxy_corr_actor_rows": proxy, "q_mean": float(q.mean()), "q_scale": float(q_scale),
            "grad_q_norm": float(jnp.linalg.norm(gq)), "grad_bc_norm": float(jnp.linalg.norm(gb)),
            "grad_ratio_q_over_bc": float(jnp.linalg.norm(gq) / jnp.linalg.norm(gb)),
            "q_std_over_u_draws_div_qscale": float(jnp.mean(jnp.std(qd, axis=1)) / q_scale),
            "q_std_over_u_draws": float(jnp.mean(jnp.std(qd, axis=1))),
            "bc_err": float(jnp.mean((u - tgt) ** 2)),
            "moments": moments(agent.z_mu, agent.z_cov)}


def check_P(agent, batch, cfg):
    c = agent.config
    agent = agent.replace(rng=jax.random.PRNGKey(1))
    batch = {**batch, **agent.reward_index_rows(batch)}
    obs = jnp.asarray(batch["observations"])
    pool = jnp.asarray(batch["next_observations"])
    w_a = jnp.asarray(batch["index_w_actor"])
    f_c = jnp.asarray(batch["index_f"]) - agent.fb_mu[None]
    view = w_a @ f_c.T / jnp.sqrt(float(c["z_dim"]))                               # (N, P)
    kern = kernel_reward(jnp.asarray(batch["fb_goals"]), pool, parse_hit_spec(c["fb_hit_spec"]))
    proxy = corr_rows(view, kern, jnp.asarray(batch["index_hindsight_actor"]))
    fa = c["fbc_actor"]
    noise = jax.random.normal(jax.random.PRNGKey(3), (obs.shape[0], int(c["action_dim"])))
    u = agent.fbc_actor_u(obs, w_a, noise)
    tgt = agent.fbc_rollout(agent.fbc_vf.params, obs, noise)
    bp = agent.basis.params
    q, _ = agent.reward_softmax_q(obs, u, pool, w_a, f_c, bp)
    q_scale = jnp.maximum(jnp.abs(q - q.mean()).mean(), 1e-6)

    def qterm(uu):
        return -float(fa["q_coeff"]) * agent.reward_softmax_q(obs, uu, pool, w_a, f_c, bp)[0].mean() / q_scale

    def bcterm(uu):
        return float(fa["bc_coeff"]) * jnp.mean((uu - tgt) ** 2)
    gq, gb = jax.grad(qterm)(u), jax.grad(bcterm)(u)
    n = N_SENS
    eps = jnp.clip(jax.random.normal(jax.random.PRNGKey(2), (n, K_DRAWS, u.shape[-1])), -3.0, 3.0)
    o = jnp.repeat(obs[:n], K_DRAWS, 0)
    ww = jnp.repeat(w_a[:n], K_DRAWS, 0)
    qd = agent.reward_softmax_q(o, eps.reshape(n * K_DRAWS, -1), pool, ww, f_c, bp)[0].reshape(n, K_DRAWS)
    # the measure's own w-dependence: share of the logit's spread over next states explained by phi^T w
    phi, b = agent.mesh_phi_b(obs[:64], u[:64], pool, head=0)
    pw = (phi * w_a[:64, None, :]).sum(-1)
    return {"proxy_corr_actor_rows": proxy, "q_mean": float(q.mean()), "q_scale": float(q_scale),
            "grad_q_norm": float(jnp.linalg.norm(gq)), "grad_bc_norm": float(jnp.linalg.norm(gb)),
            "grad_ratio_q_over_bc": float(jnp.linalg.norm(gq) / jnp.linalg.norm(gb)),
            "q_std_over_u_draws_div_qscale": float(jnp.mean(jnp.std(qd, axis=1)) / q_scale),
            "q_std_over_u_draws": float(jnp.mean(jnp.std(qd, axis=1))),
            "bc_err": float(jnp.mean((u - tgt) ** 2)),
            "logit_std_over_next_states_phiw": float(jnp.mean(jnp.std(pw, axis=1))),
            "logit_std_over_next_states_b": float(jnp.mean(jnp.std(b, axis=1))),
            "moments": moments(agent.fb_mu, agent.fb_cov)}


def main():
    out = sys.argv[1]
    _, _, train, _ = make_env_and_datasets("cube-single-play-singletask-v0")
    base = dict(train)
    aug, _ = repair_invalid_preimages(load_augmented_dataset(PRE))
    for k in aug:
        if k.startswith("noise_preimage"):
            base[k] = aug[k]
    runs = [("L_bc0p3", sorted(glob.glob(f"{EXP}/dfb_latent_oc_bc0p3_cube/*"))[0], 500000, check_L)]
    runs += [(f"P_bc03_{os.path.basename(d)[:5]}", d, 500000, check_P)
             for d in sorted(glob.glob(f"{EXP}/psmgoal_recipe_bc03_cube/*"))[:1]]
    res = {}
    for name, d, ep, fn in runs:
        agent, ds, cfg = load(d, ep, base)
        np.random.seed(0)
        batch = {k: jnp.asarray(v) for k, v in ds.sample(int(cfg["batch_size"])).items()}
        res[name] = fn(agent, batch, cfg)
        res[name]["run"] = d
        res[name]["batch_size"] = int(cfg["batch_size"])
        print(name, json.dumps(res[name], indent=1), flush=True)
        if name.startswith("L"):
            # the same L checkpoint on a 512-row batch (P's batch size)
            np.random.seed(0)
            b2 = {k: jnp.asarray(v) for k, v in ds.sample(512).items()}
            res[name + "_batch512"] = fn(agent, b2, cfg)
            print(name + "_batch512", json.dumps(res[name + "_batch512"], indent=1), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
