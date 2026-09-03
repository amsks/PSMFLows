"""Re-evaluate a saved checkpoint at high episode count, with a proper interval.

Why this exists: in-loop eval runs `eval_episodes` (50 in our Stage-C launches), which
puts a 95% CI of about +/-0.115 on any single success number -- wide enough that the
whole 0.22..0.36 band a run wanders through is one flat line plus noise. Comparing a
Stage-C score against its own behavior-cloning prior needs both sides measured tightly,
so this reloads weights and evaluates with N episodes and no training.

Two agents matter here and both work unchanged:
  * psmflow -- needs the reward-inferred task vector, so `infer_eval_z` is applied
    exactly as main.py's eval block does (same relabel size, same reward shift).
  * fql with bc_only -- `sample_actions` draws a fresh N(0, I) latent every step and
    decodes it, i.e. running the frozen Stage-A flow IS the per-step-prior BC control.
    No z inference, no preimages.

The agent config is taken from the RUN'S OWN flags.json (under restore_path) wherever
one exists, with anything typed on the command line layered on top. Before 2026-09-03 it
came from the hydra `agent` group plus CLI overrides only, so a run trained off-default
(`agent.policy_index=latent`, `agent.train_actor=false`, a non-default `agent.u_clip` or
`agent.acting`, latentrl's `agent.critic_input`) silently evaluated a DIFFERENT policy
unless every flag was re-typed on the eval line. `restore_agent` replaces the parameter
tree wholesale without a shape check, so that failure is loud for a width change and
silent for `u_clip` (the actor is tanh * u_clip: a 3x action scale, no error).

Reports mean success with a Wilson 95% interval (the normal approximation misbehaves
near 0 and 1, and the BC control could land anywhere), plus the per-episode successes
so several runs can be pooled later.

Run: MUJOCO_GL=egl .venv/bin/python tools/eval_checkpoint.py \
    agent=psmflow agent.flow_ckpt_path=<flow_dir> agent.flow_ckpt_epoch=500000 \
    agent.preimage_path=<npz> env_name=cube-single-play-singletask-v0 \
    restore_path=<run_dir> restore_epoch=500000 eval_episodes=500
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see module docstring)

import hydra
import ml_collections
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from agents import agents
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.datasets import Dataset
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent
from utils.log_utils import write_report


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(centre - half, 4), round(centre + half, 4))


def _cli_agent_keys():
    """The `agent.*` keys the caller actually typed, so they can be re-applied last.

    Hydra records the raw override strings; `agent=psmflow` selects the group and is not
    a value, so only dotted `agent.<...>` entries count. An explicit override must beat
    the run's flags.json -- that is what makes a deliberate off-config eval still possible.
    """
    try:
        task = HydraConfig.get().overrides.task
    except ValueError:  # not inside a hydra job (unit tests call the helpers directly)
        return set()
    keys = set()
    for raw in task:
        item = str(raw).lstrip("+~")
        if "=" not in item:
            continue
        key = item.split("=", 1)[0].strip()
        if key.startswith("agent."):
            keys.add(key[len("agent."):])
    return keys


def _get_path(d, path):
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            raise KeyError(path)
        d = d[part]
    return d


def _set_path(d, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        d = d[part]
    d[parts[-1]] = value


def merge_run_config(cli_agent, restore_path, cli_keys):
    """Run's flags.json as the DEFAULTS, the typed CLI overrides on top.

    Only keys present in BOTH dicts are inherited: the schema stays exactly the one this
    checkout builds agents from, so a flags.json from an older or newer config cannot add
    or remove a field -- it can only change a value this code already reads. Returns
    (merged dict, provenance dict) and is a pure function so it is unit-testable without
    hydra, an env or a GPU.
    """
    prov = {"flags_json": None, "inherited": {}, "cli_overrides": sorted(cli_keys),
            "ignored_run_only_keys": []}
    if not restore_path:
        return cli_agent, prov
    path = os.path.join(str(restore_path), "flags.json")
    if not os.path.exists(path):
        return cli_agent, prov
    try:
        with open(path) as fh:
            run_flags = json.load(fh)
    except (OSError, ValueError) as e:
        print(f"WARNING: could not read {path} ({e}); using the hydra agent config only")
        return cli_agent, prov
    run_agent = run_flags.get("agent")
    if not isinstance(run_agent, dict):
        return cli_agent, prov
    prov["flags_json"] = path
    # A checkpoint from a different agent cannot be restored into this one anyway; fail
    # here with the reason rather than inside restore_agent's tree replacement.
    assert run_agent.get("agent_name") == cli_agent.get("agent_name"), (
        f"{path} was written by agent={run_agent.get('agent_name')!r} but this eval builds "
        f"agent={cli_agent.get('agent_name')!r}")

    merged = json.loads(json.dumps(cli_agent))  # deep copy, plain containers

    def walk(run_d, out_d, prefix=""):
        for k, v in run_d.items():
            if k not in out_d:
                prov["ignored_run_only_keys"].append(prefix + k)
                continue
            if isinstance(v, dict) and isinstance(out_d[k], dict):
                walk(v, out_d[k], prefix + k + ".")
                continue
            if out_d[k] != v:
                prov["inherited"][prefix + k] = {"config": out_d[k], "run": v}
                out_d[k] = v

    walk(run_agent, merged)
    # The typed overrides win, re-applied after the inherit pass.
    for key in cli_keys:
        try:
            value = _get_path(cli_agent, key)
        except KeyError:
            continue
        try:
            _set_path(merged, key, value)
        except (KeyError, TypeError):
            continue
        prov["inherited"].pop(key, None)
    return merged, prov


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    # Dataset.sample() draws from the GLOBAL numpy stream, so the task-vector relabel
    # batch below (and anything else sampling here) was entropy-dependent: two runs of
    # this tool on the same checkpoint inferred w from different transitions. Pin it.
    np.random.seed(int(cfg.seed))

    env, eval_env, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    cli_agent = OmegaConf.to_container(cfg.agent, resolve=True)
    merged, prov = merge_run_config(cli_agent, cfg.restore_path, _cli_agent_keys())
    if prov["flags_json"]:
        print(f"agent config defaults from {prov['flags_json']}")
        for k, d in sorted(prov["inherited"].items()):
            print(f"  {k}: config {d['config']!r} -> run {d['run']!r}")
        if prov["cli_overrides"]:
            print(f"  CLI overrides kept: {', '.join(prov['cli_overrides'])}")
        if not prov["inherited"]:
            print("  (run config already matches this checkout's defaults)")
    else:
        print(f"NOTE: no flags.json under restore_path={cfg.restore_path!r}; "
              "agent config is the hydra group plus CLI overrides only")
    config = ml_collections.ConfigDict(_lists_to_tuples(merged))
    name = config["agent_name"]

    ex = ds.sample(1)
    agent = agents[name].create(cfg.seed, ex["observations"], ex["actions"], config)
    assert cfg.restore_path is not None, "needs a trained checkpoint (restore_path)"
    agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    # Task vector, identical to main.py's eval block. fql/bc_only has no infer_eval_z
    # and acts straight off observations.
    if hasattr(agent, "infer_eval_z"):
        n_relabel = min(ds.size, int(cfg.get("eval_relabel_size", 10000)))
        zb = ds.sample(n_relabel)
        shift = float(cfg.get("eval_reward_shift", 1.0))
        agent = agent.infer_eval_z(zb["next_observations"], zb["rewards"] + shift)

    n_ep = int(cfg.eval_episodes)
    # Same seed convention as training eval: the episode-init stream starts where the
    # in-loop eval's does, and (since the 08-14 seeding fix) the action-noise stream is
    # pinned to cfg.seed too, so re-running this tool on a checkpoint reproduces its
    # number exactly. It does NOT reproduce an in-loop eval step-for-step -- the in-loop
    # relabel batch is drawn from a global numpy stream that training has advanced --
    # which is what the previous docstring claimed.
    info, trajs, _ = evaluate(agent=agent, env=eval_env, config=config,
                              num_eval_episodes=n_ep, num_video_episodes=0,
                              seed=int(cfg.seed))

    # `evaluate` averages info fields; recover per-episode successes from the
    # trajectories so the interval is computed on counts, not on a mean.
    per_ep = [float(np.max(np.asarray(t["info"][-1].get("success", 0.0)))) if "info" in t
              else None for t in trajs]
    per_ep = [p for p in per_ep if p is not None]
    if per_ep:
        k, n = int(sum(p > 0.5 for p in per_ep)), len(per_ep)
    else:  # fall back to the averaged field if the env does not expose per-step success
        k, n = int(round(float(info["success"]) * n_ep)), n_ep

    # Self-identifying metadata. The same checkpoint evaluated in three acting modes
    # produced three JSONs distinguishable only by their FILENAME, which is exactly how a
    # lambda-rank number gets quoted as the deployed one six weeks later. Record what was
    # actually run, in the report.
    ac = config.get("action_critic", {}) or {}
    rank_k = int(ac.get("eval_rank_k", 0) or 0)
    if not ac.get("enabled", False):
        mode = "decode(actor latent) — action branch disabled"
    elif rank_k == 1:
        mode = "decode-only control: one actor draw, decoded, no residual, no selection"
    elif rank_k > 1:
        mode = f"lambda-rank: {rank_k} decoded candidates scored by Q_a, argmax, no residual"
    else:
        mode = (f"deployed: actor draw + eps-bounded residual "
                f"(residual_eps={config.get('residual_eps')})")

    lo, hi = wilson(k, n)
    report = {
        "env": cfg.env_name,
        "agent": name,
        "acting": config.get("acting"),
        "acting_mode": mode,
        "action_critic": {"enabled": bool(ac.get("enabled", False)),
                          "eval_rank_k": rank_k,
                          "residual_eps": config.get("residual_eps"),
                          "fb_graft": bool(ac.get("fb_graft", False))},
        "dataset_fraction": float(cfg.get("dataset_fraction", 1.0)),
        "dataset_fraction_seed": int(cfg.get("dataset_fraction_seed", 0)),
        # latentrl's two must-repeat flags (see scripts/eval500.sh's header): the actor is
        # tanh * u_clip and the critic's input space is a config switch, so a JSON that does
        # not record them cannot be checked against the run's flags.json after the fact.
        "u_clip": config.get("u_clip"),
        "critic_input": config.get("critic_input"),
        "flow_ckpt_path": str(config.get("flow_ckpt_path")),
        "preimage_path": str(config.get("preimage_path")),
        # Where each agent-config value came from, so a JSON can be checked against the
        # run's flags.json after the fact instead of trusted.
        "agent_config_source": prov,
        "policy_index": config.get("policy_index"),
        "train_actor": config.get("train_actor"),
        "restore_path": str(cfg.restore_path),
        "restore_epoch": int(cfg.restore_epoch),
        "seed": int(cfg.seed),
        "num_episodes": n,
        "num_success": k,
        "success": round(k / n, 4),
        "wilson95": [lo, hi],
        "half_width": round((hi - lo) / 2, 4),
        "eval_success_field": round(float(info["success"]), 4),
        "per_episode_success": [int(p > 0.5) for p in per_ep],
    }
    print(f"\n{name} [{cfg.env_name}] {k}/{n} = {report['success']:.3f}  "
          f"95% CI [{lo:.3f}, {hi:.3f}]\n  mode: {mode}")
    write_report(report, cfg, "eval_checkpoint.json")


if __name__ == "__main__":
    main()
