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

Parallel episodes (`eval_workers=N`, or EVAL_WORKERS in the environment)
------------------------------------------------------------------------
The rollout is MuJoCo stepping on ONE cpu core with a batch-1 flow decode per step: a
500-episode antmaze eval is ~95 min of wall clock at 42% of a core and ~3.2 GB of the
H100's 80 GB. N worker PROCESSES therefore fit on one GPU and finish in ~1/N the time.

Each worker builds its own env, restores the same checkpoint, and runs a contiguous
shard of the episodes; the parent aggregates into the same JSON. Seeding:

  * `seed` (cfg.seed) is the EVAL seed, unchanged in meaning.
  * worker w of N uses `worker_seed = seed * N + w` -- so N=1 is `seed` itself and the
    run is BIT-IDENTICAL to the pre-2026-09-07 single-process tool, while two evals of
    the same checkpoint at different `seed` never share an episode-init stream.
  * the task vector is inferred from `np.random.seed(seed)` in every worker, i.e. z is
    the same z the single-process tool infers, not a per-worker one.

`utils.evaluation.evaluate` seeds the env's episode-init RNG ONCE per call and then lets
per-episode resets advance it, and draws its action keys from one sequential stream, so
episode i of a 500-episode run cannot be reproduced by a worker that did not run
episodes 0..i-1. N>1 is therefore a DIFFERENT sample of 500 episode inits, not a
re-ordering of the same ones -- agreement with an N=1 number is agreement within the
Wilson interval, not equality. `num_workers`, `worker_seeds` and `worker_episodes` are
recorded in the report so any number can be traced back to how it was split.
"""
import json
import math
import multiprocessing as mp
import os
import sys
import time

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


# --- parallel-episode plumbing -------------------------------------------------------
# Pure functions: no jax, no env, no GPU. tests/test_eval_workers.py covers them.

def resolve_num_workers(cfg_value, env_value, num_episodes):
    """How many worker processes to split `num_episodes` over.

    Precedence: an explicit `eval_workers=N` on the hydra line (anything but the config
    default of 1) beats EVAL_WORKERS in the environment, which beats 1. Clamped to
    [1, num_episodes] -- a worker with zero episodes would build an env and a flow for
    nothing, and `evaluate` on 0 episodes returns an empty stats dict.
    """
    n = None
    if cfg_value is not None:
        try:
            cfg_n = int(cfg_value)
        except (TypeError, ValueError):
            cfg_n = 1
        if cfg_n != 1:
            n = cfg_n
    if n is None and env_value not in (None, ""):
        try:
            n = int(env_value)
        except (TypeError, ValueError):
            n = None
    if n is None:
        n = 1
    return max(1, min(int(n), int(num_episodes)))


def split_episodes(num_episodes, num_workers):
    """Contiguous, near-equal shards; the remainder goes to the LOW worker indices.

    Deterministic given (num_episodes, num_workers), which is what makes
    `per_episode_success` reproducible: worker w always owns the same COUNT, and the
    report concatenates the shards in worker order.
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    if num_episodes < num_workers:
        raise ValueError(f"cannot split {num_episodes} episodes over {num_workers} workers")
    base, rem = divmod(int(num_episodes), int(num_workers))
    return [base + (1 if w < rem else 0) for w in range(int(num_workers))]


def worker_seed_plan(base_seed, num_workers):
    """Per-worker eval seeds: `base_seed * num_workers + w`.

    Two properties this buys, both load-bearing:
      * N=1 gives back `base_seed` exactly, so the single-process path is unchanged and
        every eval500 JSON recorded before 2026-09-07 stays reproducible.
      * seeds are disjoint across `base_seed` at a fixed N, so an eval at seed 0 and one
        at seed 1 never draw the same episode inits.
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    return [int(base_seed) * int(num_workers) + w for w in range(int(num_workers))]


def aggregate_worker_results(results):
    """Fold the per-worker shard dicts into the counts the report is built from.

    `results` is a list of dicts with keys worker / seed / num_episodes /
    per_episode_raw / stats_success. Returns
    (per_episode_raw_in_worker_order, num_success, num_episodes, eval_success_field).

    Two shapes have to survive: an env that exposes a per-step `success` (every OGBench
    env we run) gives per-episode values and the interval is computed on counts; an env
    that does not gives only `evaluate`'s averaged field, and the count is recovered from
    the EPISODE-WEIGHTED mean -- an unweighted mean over workers would be wrong the
    moment 500 does not divide evenly by N.
    """
    ordered = sorted(results, key=lambda r: int(r["worker"]))
    expected = list(range(len(ordered)))
    got = [int(r["worker"]) for r in ordered]
    if got != expected:
        raise ValueError(f"worker indices {got} are not a contiguous 0..{len(ordered) - 1}")
    for r in ordered:
        n_claimed, n_raw = int(r["num_episodes"]), len(r["per_episode_raw"])
        if n_raw and n_raw != n_claimed:
            raise ValueError(f"worker {r['worker']} was given {n_claimed} episodes but "
                             f"returned {n_raw} per-episode values")
    per_ep = [float(p) for r in ordered for p in r["per_episode_raw"]]
    total = sum(int(r["num_episodes"]) for r in ordered)
    if per_ep:
        return per_ep, int(sum(p > 0.5 for p in per_ep)), len(per_ep), float(np.mean(per_ep))
    mean = sum(float(r["stats_success"]) * int(r["num_episodes"]) for r in ordered) / total
    return [], round(mean * total), total, float(mean)


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


# Agent defaults that CHANGED when the affine, paper-strict LatentFlowPSM became THE
# psmflow default (2026-09-04). A flags.json written before a key existed does not carry it
# at all, so the inherit pass below would leave this checkout's NEW default in place and
# silently evaluate an OLD checkpoint as if it were the new agent -- and for `psi_form` it
# would trip create()'s policy_index guard outright. Rule: key absent from the run's own
# flags.json => the value that was the default when that run was written.
LEGACY_AGENT_DEFAULTS = {
    "psmflow": {
        "psi_form": "free",
        "policy_index": "task_vector",
        "acting": "actor",
        "train_actor": True,
        "use_point_preimage": False,
    },
}


def merge_run_config(cli_agent, restore_path, cli_keys):
    """Run's flags.json as the DEFAULTS, the typed CLI overrides on top.

    Only keys present in BOTH dicts are inherited: the schema stays exactly the one this
    checkout builds agents from, so a flags.json from an older or newer config cannot add
    or remove a field -- it can only change a value this code already reads. Returns
    (merged dict, provenance dict) and is a pure function so it is unit-testable without
    hydra, an env or a GPU.
    """
    prov = {"flags_json": None, "inherited": {}, "cli_overrides": sorted(cli_keys),
            "ignored_run_only_keys": [], "legacy_defaults": {}, "train_seed": None}
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
    # The TRAINING seed, which the report otherwise has no record of: `seed` there is the
    # EVAL seed (cfg.seed, 0 for every eval500 ever run), so three seeds of one arm produced
    # three JSONs whose only trace of which seed they came from was `restore_path` and the
    # filename. Pooling across seeds has to be checkable from the JSON itself.
    if isinstance(run_flags.get("seed"), (int, float)):
        prov["train_seed"] = int(run_flags["seed"])
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
    # Pre-flip checkpoints: a key this checkout has but the run's flags.json does not is a
    # key that did not exist when the run was written, so its value there is the OLD default
    # -- not this checkout's. Restore it, and say so in the provenance.
    for k, legacy in LEGACY_AGENT_DEFAULTS.get(cli_agent.get("agent_name"), {}).items():
        if k not in run_agent and k in merged and merged[k] != legacy:
            prov["legacy_defaults"][k] = {"config": merged[k], "legacy": legacy}
            merged[k] = legacy
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
        prov["legacy_defaults"].pop(key, None)
    return merged, prov


def _evaluate_shard(payload):
    """One worker: build env + agent, run this shard's episodes, return raw successes.

    Runs in a SPAWNED child (never a fork: jax/CUDA and MuJoCo's EGL context do not
    survive one), so this module is re-imported from scratch there -- which is exactly
    what keeps `utils.xla_guard` ahead of jax, since it is the module's first import.
    Everything in `payload` is plain json-able containers; nothing jax-typed crosses the
    process boundary.

    N=1 calls this in-process, and the sequence below is the pre-2026-09-07 body of
    `main` verbatim (same global-seed point, same sample order, same `evaluate` call), so
    a single-worker run reproduces old numbers bit for bit.
    """
    idx = int(payload["worker"])
    if idx > 0:  # one progress bar (worker 0's) is informative; eight interleaved are not
        os.environ.setdefault("TQDM_DISABLE", "1")
    base_seed = int(payload["base_seed"])
    # Dataset.sample() draws from the GLOBAL numpy stream; pinning it to the BASE seed
    # (not the worker seed) is what makes every worker infer the SAME task vector z, and
    # the same one the single-process tool infers.
    np.random.seed(base_seed)

    _, eval_env, train_dataset, _ = make_env_and_datasets(
        payload["env_name"], frame_stack=payload["frame_stack"])
    ds = Dataset.create(**train_dataset)
    config = ml_collections.ConfigDict(_lists_to_tuples(payload["agent_config"]))
    name = config["agent_name"]

    ex = ds.sample(1)
    agent = agents[name].create(base_seed, ex["observations"], ex["actions"], config)
    agent = restore_agent(agent, payload["restore_path"], payload["restore_epoch"])

    # Task vector, identical to main.py's eval block. fql/bc_only has no infer_eval_z
    # and acts straight off observations.
    if hasattr(agent, "infer_eval_z"):
        n_relabel = min(ds.size, int(payload["relabel_size"]))
        zb = ds.sample(n_relabel)
        agent = agent.infer_eval_z(zb["next_observations"], zb["rewards"] + payload["reward_shift"])

    t0 = time.time()
    info, trajs, _ = evaluate(agent=agent, env=eval_env, config=config,
                              num_eval_episodes=int(payload["num_episodes"]),
                              num_video_episodes=0, seed=int(payload["seed"]))
    # `evaluate` averages info fields; recover per-episode successes from the
    # trajectories so the interval is computed on counts, not on a mean.
    per_ep = [float(np.max(np.asarray(t["info"][-1].get("success", 0.0)))) if "info" in t
              else None for t in trajs]
    per_ep = [p for p in per_ep if p is not None]
    return {
        "worker": idx,
        "seed": int(payload["seed"]),
        "num_episodes": int(payload["num_episodes"]),
        "per_episode_raw": per_ep,
        "stats_success": float(info["success"]) if "success" in info else 0.0,
        "seconds": round(time.time() - t0, 1),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    cli_agent = OmegaConf.to_container(cfg.agent, resolve=True)
    merged, prov = merge_run_config(cli_agent, cfg.restore_path, _cli_agent_keys())
    if prov["flags_json"]:
        print(f"agent config defaults from {prov['flags_json']}")
        for k, d in sorted(prov["inherited"].items()):
            print(f"  {k}: config {d['config']!r} -> run {d['run']!r}")
        for k, d in sorted(prov["legacy_defaults"].items()):
            print(f"  {k}: absent from flags.json -> pre-2026-09-04 default "
                  f"{d['legacy']!r} (not this checkout's {d['config']!r})")
        if prov["cli_overrides"]:
            print(f"  CLI overrides kept: {', '.join(prov['cli_overrides'])}")
        if not prov["inherited"] and not prov["legacy_defaults"]:
            print("  (run config already matches this checkout's defaults)")
    else:
        print(f"NOTE: no flags.json under restore_path={cfg.restore_path!r}; "
              "agent config is the hydra group plus CLI overrides only")
    config = ml_collections.ConfigDict(_lists_to_tuples(merged))
    name = config["agent_name"]
    assert cfg.restore_path is not None, "needs a trained checkpoint (restore_path)"

    n_ep = int(cfg.eval_episodes)
    base_seed = int(cfg.seed)
    n_workers = resolve_num_workers(cfg.get("eval_workers", 1),
                                    os.environ.get("EVAL_WORKERS"), n_ep)
    shards = split_episodes(n_ep, n_workers)
    seeds = worker_seed_plan(base_seed, n_workers)
    payloads = [{
        "worker": w,
        "seed": seeds[w],
        "base_seed": base_seed,
        "num_episodes": shards[w],
        "env_name": cfg.env_name,
        "frame_stack": cfg.frame_stack,
        "agent_config": merged,
        "restore_path": str(cfg.restore_path),
        "restore_epoch": cfg.restore_epoch,
        "relabel_size": int(cfg.get("eval_relabel_size", 10000)),
        "reward_shift": float(cfg.get("eval_reward_shift", 1.0)),
    } for w in range(n_workers)]

    t_start = time.time()
    if n_workers == 1:
        # In-process, and byte-identical to the pre-2026-09-07 tool: same global seed
        # point, same sample order, `evaluate(seed=cfg.seed)`.
        results = [_evaluate_shard(payloads[0])]
        mem_frac = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
    else:
        # Divide the device memory pool BEFORE the children are spawned: XLA reads this
        # when the backend initializes, and the child inherits the parent's environ. The
        # parent must not touch a jax device here -- it never builds an agent, so it
        # does not. Measured 2026-09-07: one eval process holds ~3.2 GB of an 80 GB H100,
        # so 0.90/N is generous for any N we would run.
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ.setdefault("MUJOCO_GL", "egl")
        whole = float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") or 0.90)
        mem_frac = f"{max(0.04, min(0.90, whole / n_workers)):.4f}"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = mem_frac
        # numpy/BLAS defaults to one thread per core on a 200-core node; N of those
        # fighting over `cpus-per-task` cores is slower than N single-threaded workers.
        per_worker_threads = max(1, len(os.sched_getaffinity(0)) // n_workers)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(var, str(per_worker_threads))
        print(f"parallel eval: {n_workers} workers, episodes {shards}, seeds {seeds}, "
              f"XLA_PYTHON_CLIENT_MEM_FRACTION={mem_frac} each")
        # spawn, never fork: a forked child inherits a half-initialized CUDA context.
        with mp.get_context("spawn").Pool(processes=n_workers) as pool:
            results = pool.map(_evaluate_shard, payloads)

    per_ep, k, n, success_field = aggregate_worker_results(results)
    wall = round(time.time() - t_start, 1)

    # Self-identifying metadata. The same checkpoint evaluated in three acting modes
    # produced three JSONs distinguishable only by their FILENAME, which is exactly how a
    # lambda-rank number gets quoted as the deployed one six weeks later. Record what was
    # actually run, in the report.
    ac = config.get("action_critic", {}) or {}
    rank_k = int(ac.get("eval_rank_k", 0) or 0)
    if not ac.get("enabled", False):
        # The latent the decode is applied to depends on `acting`: gpi (the default since
        # 2026-09-04) picks it by a per-step argmax over (u_i, u\'_j) pairs, `actor` reads it
        # off the amortized latent actor. Saying "actor latent" for both was wrong.
        # `gpi_select` (2026-09-05) chooses WHICH of the gpi_num_u draws is decoded;
        # `argmax` is the shipped rule. An ablation eval differs from the deployed one by
        # this string alone, so it has to be in the report, not only in the filename.
        sel = str(config.get("gpi_select", "argmax"))
        src = ((f"gpi {sel} over (u, u') pairs, K={config.get('gpi_num_u')}")
               if config.get("acting") == "gpi"
               else f"amortized actor latent [{config.get('actor_mode', 'ddpg')}]")
        mode = f"decode({src}) — action branch disabled"
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
        "index_clip": config.get("index_clip"),
        "critic_input": config.get("critic_input"),
        "flow_ckpt_path": str(config.get("flow_ckpt_path")),
        "preimage_path": str(config.get("preimage_path")),
        # Where each agent-config value came from, so a JSON can be checked against the
        # run's flags.json after the fact instead of trusted.
        "agent_config_source": prov,
        "gpi_select": config.get("gpi_select"),
        "gpi_num_u": config.get("gpi_num_u"),
        "gpi_topm": config.get("gpi_topm"),
        "gpi_index_seed": config.get("gpi_index_seed"),
        "policy_index": config.get("policy_index"),
        "train_actor": config.get("train_actor"),
        # Added 2026-09-06: three switches that change WHICH policy is being evaluated and
        # were previously recoverable only from the run's flags.json. `psi_form` picks the
        # measure head, `index_agg` picks how its policy slot is aggregated, and
        # `actor_mode` picks which latent actor `acting=actor` deploys.
        "psi_form": config.get("psi_form"),
        "index_agg": config.get("index_agg"),
        "actor_mode": config.get("actor_mode"),
        "actor_index_panel": (config.get("actor", {}) or {}).get("index_panel"),
        "restore_path": str(cfg.restore_path),
        "restore_epoch": int(cfg.restore_epoch),
        # `seed` is the EVAL seed (episode inits + action noise); `train_seed` is the seed
        # the restored checkpoint was TRAINED with, read from its own flags.json. They are
        # different numbers and only the first was ever recorded.
        "seed": int(cfg.seed),
        "train_seed": prov.get("train_seed"),
        "num_episodes": n,
        "num_success": k,
        "success": round(k / n, 4),
        "wilson95": [lo, hi],
        "half_width": round((hi - lo) / 2, 4),
        "eval_success_field": round(success_field, 4),
        # How the 500 episodes were split. N=1 is the historical single-process run;
        # N>1 draws a different (equally valid) sample of episode inits -- worker w runs
        # `worker_episodes[w]` episodes seeded `worker_seeds[w] = seed * N + w` -- so a
        # comparison against an N=1 number is a CI overlap, not an equality. See the
        # module docstring.
        "num_workers": n_workers,
        "worker_episodes": shards,
        "worker_seeds": seeds,
        "worker_seconds": [r["seconds"] for r in sorted(results, key=lambda r: r["worker"])],
        "eval_seed_scheme": ("single process, evaluate(seed=cfg.seed)" if n_workers == 1
                             else "worker w of N: evaluate(seed=cfg.seed * N + w); "
                                  "z inferred from np.random.seed(cfg.seed) in every worker"),
        "xla_mem_fraction_per_worker": mem_frac,
        "wall_seconds": wall,
        "per_episode_success": [int(p > 0.5) for p in per_ep],
    }
    print(f"\n{name} [{cfg.env_name}] {k}/{n} = {report['success']:.3f}  "
          f"95% CI [{lo:.3f}, {hi:.3f}]\n  mode: {mode}\n"
          f"  {n_workers} worker(s), {wall:.0f} s wall")
    write_report(report, cfg, "eval_checkpoint.json")


if __name__ == "__main__":
    main()
