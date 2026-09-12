"""Do a policy's consecutive actions pull the same way, and how far does it get?

Started as the pointmaze exact-zero bug hunt (below) and generalised when its check 3 turned
up something that is not pointmaze-specific: a policy can emit FULL-SIZE actions and displace
an order of magnitude less than random latent sampling, because a per-step argmax over a
roster redrawn every step reverses direction on most steps. The actions cancel. That is
invisible in a success number and obvious in one cosine.

Two arms per checkpoint, same rollout harness:

  deployed  the run's OWN acting rule, whatever its flags.json says -- `acting=gpi` gives the
            per-step pair-scan argmax, `acting=actor` gives the trained latent actor at
            temperature 0. One arm covers both deployments, so the comparison is fair.
  bc        the frozen flow on a fresh clipped prior latent each step. The BC control, and
            also the GPI-free control.

Reported per arm: success, displacement over an episode, action norm, the fraction of action
components pinned at the clip and of latents at u_clip, and the mean cosine between
CONSECUTIVE executed actions. Measured on pointmaze sd0@500k: bc +0.106 against the deployed
arm's -0.367, with displacement 12.4 against 0.94.

--- the original question, answered 2026-09-09 ---

2026-09-09. `affine_strict_pointmaze` reads 0.000 at 500 episodes on all three seeds and all
six late checkpoints, while its `phi` is the most expressive of the three environments
(held-out reward R^2 0.395 against cube's 0.104) and whitening the reward inference does not
move it by a single episode. An exact zero everywhere, with a healthy reward channel, is a
PIPELINE signature rather than a learning one.

This is a bug hunt, not a hypothesis. Four checks, in the order agreed with the oversight
session, stopping at the first that fails:

  1. BC control      roll the FROZEN flow on a fresh prior latent each step. If that is also
                     ~0 the flow or the eval task is broken and the agent is exonerated.
                     (This is also check 4's GPI-free control -- they are the same rollout,
                     so they are run once.)
  2. saturation      decoded action norm, fraction of action components pinned at the clip,
                     fraction of ACTING latents at u_clip. Pinned actions point at the
                     XLA/unroll guard or at clipping, not at RL.
  3. does it move    displacement over an episode, agent against BC, plus the reward density
                     the relabelling batch actually sees.
  4. GPI-free        covered by 1.

VERDICT: check 1 failed. BC scores 0/100, Wilson upper bound 0.037 -- the frozen flow cannot
do pointmaze at all, so every Stage-C number there is capped by the substrate and the agent
is exonerated. Check 2 passed (nothing pinned). Check 3 failed and is the finding above.

Run (CPU is fine; the `bc` arm is cheap, a gpi run pays a K x K pair scan per step):
  MUJOCO_GL=egl .venv/bin/python tools/diag_action_coherence.py \
      --run $PSM_DATA/exp/PSMFLows/affine_strict_cube/sd000_* --epoch 350000 \
      --flow $PSM_DATA/flow/cube-single-play \
      --env cube-single-play-singletask-v0 --episodes 100 --bc_episodes 100 \
      --out $PSM_DATA/logs/diag_action_coherence_cube_sd000_350000.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax


def _q(x):
    x = np.asarray(x, np.float64).ravel()
    return {"mean": round(float(x.mean()), 4), "p10": round(float(np.quantile(x, 0.1)), 4),
            "p90": round(float(np.quantile(x, 0.9)), 4), "max": round(float(x.max()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epoch", type=int, default=500000)
    ap.add_argument("--flow", required=True)
    ap.add_argument("--flow_epoch", type=int, default=500000)
    ap.add_argument("--preimages", default="")
    ap.add_argument("--env", default="pointmaze-medium-navigate-singletask-task1-v0")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--bc_episodes", type=int, default=0,
                    help="0 = same as --episodes. The BC arm is cheap (no pair scan), so it "
                         "can carry the episode count that settles whether it is really 0.")
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import ml_collections

    from agents import agents
    from agents.psmflow import get_config
    from envs.env_utils import make_env_and_datasets
    from main import _lists_to_tuples
    from tools.eval_checkpoint import _cli_agent_keys, merge_run_config
    from utils.datasets import Dataset
    from utils.flax_utils import restore_agent

    np.random.seed(a.seed)
    _, eval_env, train_dataset, _ = make_env_and_datasets(a.env, frame_stack=None)
    ds = Dataset.create(**train_dataset)
    ex = ds.sample(1)

    base = json.loads(json.dumps(get_config().to_dict()))
    base["flow_ckpt_path"], base["flow_ckpt_epoch"] = a.flow, a.flow_epoch
    base["preimage_path"] = a.preimages or None
    merged, prov = merge_run_config({"agent_name": "psmflow", **base}, a.run, _cli_agent_keys())
    cfg = ml_collections.ConfigDict(_lists_to_tuples(merged))
    agent = agents["psmflow"].create(a.seed, ex["observations"], ex["actions"], cfg)
    agent = restore_agent(agent, a.run, a.epoch)
    np.random.seed(a.seed)
    zb = ds.sample(min(ds.size, 10000))
    agent = agent.infer_eval_z(zb["next_observations"], zb["rewards"] + 1.0)

    ac = agent.config
    d_a, u_clip = int(ac["action_dim"]), float(ac["u_clip"])
    r = np.asarray(ds["rewards"]).ravel()
    out = {"probe": "pointmaze exact-zero bug hunt", "env": a.env, "run": a.run,
           "epoch": a.epoch, "episodes": a.episodes, "max_steps": a.max_steps,
           "agent_config_source": prov,
           "dataset": {"n": len(r), "reward_min": float(r.min()),
                       "reward_max": float(r.max()),
                       "frac_rewarding_rows": float((r > r.min()).mean())},
           "checks": {}}

    @jax.jit
    def act_agent(obs1, key):
        """The run's OWN acting rule. `sample_actions` dispatches on its `acting` flag, so
        this is the pair-scan argmax on a gpi run and the trained latent actor on an actor
        run -- `temperature=0` being the mode, which is how the eval harness deploys it.

        The latent returned alongside is only for the u_clip diagnostic.
        """
        a_out = agent.sample_actions(obs1, seed=key, temperature=0)
        u = (agent.gpi_select(obs1, seed=key) if agent.config["acting"] == "gpi"
             else agent._deploy_latent(obs1[None], agent._actor_w(agent.task_z)[None],
                                       jax.random.normal(key, (1, d_a)))[0])
        return a_out, u

    @jax.jit
    def act_bc(obs1, key):
        u = jnp.clip(jax.random.normal(key, (d_a,)), -u_clip, u_clip)
        return agent.decode(obs1[None], u[None])[0], u

    def rollout(fn, tag, n_eps):
        rng = jax.random.PRNGKey(a.seed)
        succ, lens, disp, anorm, aclip, uclip_frac, ends = [], [], [], [], [], [], []
        cos_consec = []
        for ep in range(n_eps):
            ob, _ = eval_env.reset()
            start = np.asarray(ob, np.float64).copy()
            done, t, s, prev = False, 0, 0.0, None
            while not done:
                rng, k = jax.random.split(rng)
                act, u = jax.device_get(fn(jnp.asarray(ob, jnp.float32), k))
                act = np.asarray(act)
                anorm.append(np.linalg.norm(act))
                aclip.append(float(np.mean(np.abs(act) >= 1.0 - 1e-6)))
                uclip_frac.append(float(np.mean(np.abs(np.asarray(u)) >= u_clip - 1e-6)))
                # Are consecutive actions pulling the same way? A policy emitting
                # full-size actions that displaces 10x less than random is not stuck, it is
                # CANCELLING -- and per-step argmax over a roster redrawn every step is the
                # obvious way to produce that.
                if prev is not None:
                    dn = np.linalg.norm(prev) * np.linalg.norm(act)
                    if dn > 1e-9:
                        cos_consec.append(float(prev @ act / dn))
                prev = act.copy()
                ob, _r, term, trunc, info = eval_env.step(np.clip(act, -1, 1))
                s = max(s, float(np.max(np.asarray(info.get("success", 0.0)))))
                done, t = bool(term or trunc), t + 1
                if t >= a.max_steps:
                    done = True
            succ.append(s)
            lens.append(t)
            fin = np.asarray(ob, np.float64)
            n = min(len(start), len(fin))
            disp.append(float(np.linalg.norm(fin[:n] - start[:n])))
            ends.append(fin[:2].tolist())
            print(f"  {tag} ep {ep:3d}  len {t:4d}  success {s:.0f}  disp {disp[-1]:.3f}",
                  flush=True)
        return {"success": float(np.mean(succ)), "n_episodes": n_eps,
                "episode_len": float(np.mean(lens)),
                "displacement": _q(disp), "action_norm": _q(anorm),
                "action_clipfrac": _q(aclip), "latent_clipfrac": _q(uclip_frac),
                "consecutive_action_cos": _q(cos_consec),
                "final_xy_sample": ends[:5]}

    print("=== check 1+4: BC control / GPI-free (frozen flow, fresh prior u each step) ===")
    out["checks"]["bc_control"] = rollout(act_bc, "bc", a.bc_episodes or a.episodes)
    print("=== checks 2+3: the deployed agent ===")
    out["checks"]["agent"] = rollout(act_agent, "agent", a.episodes)

    bc, ag = out["checks"]["bc_control"], out["checks"]["agent"]
    out["coherence"] = {
        "bc_consecutive_cos": bc["consecutive_action_cos"]["mean"],
        "deployed_consecutive_cos": ag["consecutive_action_cos"]["mean"],
        "displacement_ratio_deployed_over_bc": round(
            ag["displacement"]["mean"] / max(bc["displacement"]["mean"], 1e-9), 4),
        "acting": str(agent.config["acting"])}
    verdict = []
    if bc["success"] <= 0.001:
        verdict.append("CHECK 1 FAILED: the BC control also scores ~0, so the flow or the "
                       "eval task is broken and the AGENT is exonerated.")
    else:
        verdict.append(f"check 1 passed: BC control scores {bc['success']:.3f}, so the flow "
                       "and the eval task work; the agent is doing something worse than "
                       "acting randomly in latent space.")
    if ag["action_clipfrac"]["mean"] > 0.9:
        verdict.append(f"CHECK 2 FAILED: {ag['action_clipfrac']['mean']:.2f} of the agent's "
                       "action components are pinned at the clip.")
    if ag["displacement"]["mean"] < 0.1 * max(bc["displacement"]["mean"], 1e-9):
        verdict.append("CHECK 3 FAILED: the agent barely moves relative to the BC control.")
    out["verdict"] = verdict
    print()
    for v in verdict:
        print(v)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"report -> {a.out}")


if __name__ == "__main__":
    main()
