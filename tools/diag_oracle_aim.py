"""E1: can an oracle aim inside the flow's reachable action set?

The whole experimental record forks on one question: is the loss a SELECTION failure
(nothing we train can rank latents) or a REACHABILITY ceiling (the frozen flow's action
set does not contain what the task needs)? This tool removes selection entirely. At
every environment step it draws the same K prior latents the deployed GPI path draws,
decodes them with the EXACT decoder, and executes the candidate closest to a frozen
FQL expert's action. Nothing is trained; nothing is learned across steps.

Three arms roll out under one harness, all with the eval-seed protocol of
`tools/eval_checkpoint.py` (env episode-init RNG and action key both pinned to cfg.seed):

  oracle_aim   argmin_k ||decode(u_k) - a*||, executed.  <- the measurement
  oracle       a* executed directly.                     <- ceiling, must land ~0.95
  random_latent  one prior latent per step, decoded.     <- floor, must land near BC 0.068

The two controls are not decoration: if either misses, the aim number means nothing.
Note the random-latent floor decodes through the 100-step ODE while the quoted BC
control 0.068 [0.049, 0.093] decodes through the one-step distilled net, so a small gap
between them is the decoder mismatch, not a harness fault.

Decode is the frozen Stage-A flow through `psmflow.decode` at gpi_decode=ode,
flow_decode_steps=100 (asserted below) -- the one-step net's 0.0886 mean decode error is
~10% of action scale and would otherwise sit inside the measurement.

Run:
  MUJOCO_GL=egl .venv/bin/python tools/diag_oracle_aim.py agent=psmflow \
      env_name=cube-single-play-singletask-v0 \
      agent.flow_ckpt_path=<flow_dir> agent.flow_ckpt_epoch=500000 \
      agent.gpi_decode=ode agent.flow_decode_steps=100 \
      +oracle_path=<fql_run_dir> +oracle_epoch=500000 \
      +oracle_k=512 +oracle_episodes=500 \
      report_out=/data-local/amsks/PSMFLows/logs/<name>.json
"""
import json
import math
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


def wilson(k, n, z=1.96):
    """Wilson score interval (same estimator as tools/eval_checkpoint.py)."""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(centre - half, 4), round(centre + half, 4))


def rollout(eval_env, act_fn, n_episodes, seed, collect=None):
    """N episodes under `act_fn(obs, key) -> (action, extras)`.

    Seeding matches utils.evaluation.evaluate: the env's episode-init RNG is pinned once
    per arm and the per-episode resets advance it, and the action key stream starts at
    PRNGKey(seed). Each arm therefore sees the SAME episode inits, so the three numbers
    are paired, not three independent draws.
    """
    eval_env.reset(seed=int(seed))
    rng = jax.random.PRNGKey(int(seed))
    successes, lengths = [], []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done, t, info = False, 0, {}
        while not done:
            rng, key = jax.random.split(rng)
            action, extras = act_fn(obs, key)
            action = np.clip(np.asarray(action), -1.0, 1.0)
            if collect is not None:
                collect(extras)
            obs, _, terminated, truncated, info = eval_env.step(action)
            t += 1
            done = terminated or truncated
        successes.append(float(np.max(np.asarray(info.get("success", 0.0)))))
        lengths.append(t)
    return successes, lengths


def summarize(successes, lengths):
    k, n = int(sum(s > 0.5 for s in successes)), len(successes)
    lo, hi = wilson(k, n)
    return {"num_episodes": n, "num_success": k, "success": round(k / n, 4),
            "wilson95": [lo, hi], "half_width": round((hi - lo) / 2, 4),
            "mean_episode_length": float(np.mean(lengths))}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    np.random.seed(int(cfg.seed))

    n_ep = int(cfg.get("oracle_episodes", 500))
    K = int(cfg.get("oracle_k", 512))
    oracle_path = cfg.get("oracle_path", None)
    assert oracle_path, "needs +oracle_path=<frozen FQL expert run dir>"
    oracle_epoch = int(cfg.get("oracle_epoch", 500000))

    _, eval_env, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    config = ml_collections.ConfigDict(_lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    assert config["agent_name"] == "psmflow", "this tool decodes through psmflow's frozen flow"
    # The exact decoder is the point of the experiment; refuse to run with the distilled net.
    assert config["gpi_decode"] == "ode" and int(config["flow_decode_steps"]) == 100, (
        "E1 measures the reachable set under EXACT decode: pass "
        "agent.gpi_decode=ode agent.flow_decode_steps=100")

    ex = ds.sample(1)
    # Only `decode` and the frozen flow params are used off this agent. psi/actor/residual
    # are freshly initialized and never called -- there is no restore_path here by design.
    flow_agent = agents["psmflow"].create(cfg.seed, ex["observations"], ex["actions"], config)

    # Frozen expert. Its config comes from the run's OWN flags.json, not from this
    # process's hydra tree, so alpha/flow_steps/widths are the ones it was trained with.
    with open(os.path.join(oracle_path, "flags.json")) as f:
        oracle_flags = json.load(f)
    oracle_cfg = ml_collections.ConfigDict(_lists_to_tuples(oracle_flags["agent"]))
    oracle = agents[oracle_cfg["agent_name"]].create(
        cfg.seed, ex["observations"], ex["actions"], oracle_cfg)
    oracle = restore_agent(oracle, oracle_path, oracle_epoch)

    # psmflow.create copies the config before filling in ob_dims/action_dim, so read the
    # dimensions off the constructed agent rather than the ConfigDict handed to it.
    adim, u_clip = int(flow_agent.config["action_dim"]), float(config["u_clip"])

    print("\n--- E1 oracle-aim hyperparameters ---")
    for k, v in [
        ("env", cfg.env_name), ("seed", int(cfg.seed)), ("episodes/arm", n_ep),
        ("K candidate latents", K), ("u_clip", u_clip),
        ("decode", f'{config["gpi_decode"]} x {config["flow_decode_steps"]}'),
        ("flow_ckpt", f'{config["flow_ckpt_path"]} @ {config["flow_ckpt_epoch"]}'),
        ("oracle", f"{oracle_path} @ {oracle_epoch}"),
        ("oracle agent", f'{oracle_cfg["agent_name"]} alpha={oracle_cfg.get("alpha")} '
                         f'flow_steps={oracle_cfg.get("flow_steps")}'),
    ]:
        print(f"  {k:22s} {v}")
    print()

    @jax.jit
    def oracle_action(obs, key):
        return jnp.clip(oracle.sample_actions(obs, seed=key), -1.0, 1.0)

    @jax.jit
    def aim_action(obs, key):
        """Best-of-K under an oracle target. The candidate draw is byte-identical in form
        to the deployed GPI draw (agents/psmflow.py gpi_select): clipped N(0, I)."""
        k_or, k_u = jax.random.split(key)
        a_star = jnp.clip(oracle.sample_actions(obs, seed=k_or), -1.0, 1.0)
        u = jnp.clip(jax.random.normal(k_u, (K, adim)), -u_clip, u_clip)
        obs_b = jnp.broadcast_to(obs, (K, *obs.shape))
        a = flow_agent.decode(obs_b, u)
        d = jnp.linalg.norm(a - a_star[None], axis=-1)
        i = jnp.argmin(d)
        return a[i], (d[i], jnp.median(d), jnp.linalg.norm(a_star))

    @jax.jit
    def random_latent_action(obs, key):
        u = jnp.clip(jax.random.normal(key, (1, adim)), -u_clip, u_clip)
        return flow_agent.decode(obs[None], u)[0]

    # Controls first: the aim number is not readable until both land.
    print(f"[1/3] oracle (ceiling, expect ~0.95): {n_ep} episodes")
    s_or, l_or = rollout(eval_env, lambda o, k: (oracle_action(o, k), None), n_ep, cfg.seed)
    r_or = summarize(s_or, l_or)
    print(f"      success {r_or['success']:.3f} {r_or['wilson95']}")

    print(f"[2/3] random_latent (floor, expect ~BC 0.068): {n_ep} episodes")
    s_rl, l_rl = rollout(eval_env, lambda o, k: (random_latent_action(o, k), None), n_ep, cfg.seed)
    r_rl = summarize(s_rl, l_rl)
    print(f"      success {r_rl['success']:.3f} {r_rl['wilson95']}")

    print(f"[3/3] oracle_aim: {n_ep} episodes x K={K} exact decodes/step")
    dmin, dmed, anorm = [], [], []

    def collect(extras):
        d0, d1, d2 = extras
        dmin.append(float(d0)); dmed.append(float(d1)); anorm.append(float(d2))

    s_aim, l_aim = rollout(eval_env, aim_action, n_ep, cfg.seed, collect=collect)
    r_aim = summarize(s_aim, l_aim)
    print(f"      success {r_aim['success']:.3f} {r_aim['wilson95']}")

    dmin, dmed, anorm = np.asarray(dmin), np.asarray(dmed), np.asarray(anorm)

    def dist_stats(x):
        return {"mean": float(x.mean()), "median": float(np.median(x)),
                "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
                "p99": float(np.percentile(x, 99))}

    aim = r_aim["success"]
    branch = ("selection: reachable set is fine, the loss is ranking" if aim >= 0.7 else
              "reachability: exact decode is the ceiling, no critic can help" if aim <= 0.3 else
              "partial ceiling: read the min-distance distribution, run the 3-seed variant")

    report = {
        "experiment": "E1 oracle-aim",
        "env": cfg.env_name,
        "seed": int(cfg.seed),
        "num_candidates_K": K,
        "u_clip": u_clip,
        "decode": {"mode": config["gpi_decode"], "steps": int(config["flow_decode_steps"])},
        "flow_ckpt_path": str(config["flow_ckpt_path"]),
        "flow_ckpt_epoch": int(config["flow_ckpt_epoch"]),
        "oracle_path": str(oracle_path),
        "oracle_epoch": oracle_epoch,
        "oracle_agent": {k: oracle_cfg.get(k) for k in ("agent_name", "alpha", "flow_steps")},
        "arms": {"oracle_aim": r_aim, "oracle": r_or, "random_latent": r_rl},
        "controls_quoted": {"bc_per_step_prior": [0.068, 0.049, 0.093],
                            "psmflow_point_actor": [0.236, 0.071],
                            "fb": [0.721, 0.020], "fql_task2": [0.949, 0.063]},
        "aim_distance": {"min_over_K": dist_stats(dmin), "median_over_K": dist_stats(dmed),
                         "oracle_action_norm": dist_stats(anorm),
                         "n_steps": int(dmin.size)},
        "fork_branch": branch,
    }
    print(f"\noracle_aim {aim:.3f} -> {branch}")
    write_report(report, cfg, "diag_oracle_aim.json")


if __name__ == "__main__":
    main()
