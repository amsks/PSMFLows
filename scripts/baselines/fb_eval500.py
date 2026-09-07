"""500-episode eval of one Factored-FB checkpoint, in PSMFlows' report format.

Why this exists rather than the sibling repo's own `scripts/reeval_checkpoint.py`:
that script loops `adapter.eval_tasks(env, cfg)`, which for `--domain cube_single` is
OGBench tasks 1..5, so a 500-episode call costs 2500 episodes and produces five rows.
PSMFlows reports `cube-single-play-singletask-v0`, i.e. OGBench cube-single TASK 1 only
(tools/eval_checkpoint.py + scripts/eval500.sh), so this evaluates ONE task_id and emits
exactly the JSON schema tools/eval_checkpoint.py writes -- success, num_success,
num_episodes, wilson95, restore_epoch -- so a baseline row can sit next to a PSMFlow row
without a units conversion.

Everything about HOW the agent is built and scored is the sibling repo's own code:
env.make_config / get_adapter / Agent.create / restore_agent / impls.utils.evaluation.
evaluate, in the same order scripts/reeval_checkpoint.py calls them, including
--config_from_run (rebuild the nets from the config.json the run recorded, not from
today's yaml).

Run:
  FB_REPO=/mnt/home/amohan/git/Austin/Factored-FB \
  .../Factored-FB/.venv/bin/python scripts/baselines/fb_eval500.py \
      --critic fb --actor ddpgbc --domain cube_single --seed 0 \
      --restore_path <run_dir> --restore_epoch 500000 --task_id 1 \
      --eval_episodes 500 --out <json>
"""
import json
import math
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _cvd.split(",")[0].strip() or "0")

FB_REPO = os.environ.get("FB_REPO", "/mnt/home/amohan/git/Austin/Factored-FB")
sys.path.insert(0, FB_REPO)

from absl import app, flags  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from impls.agents import Agent  # noqa: E402
from impls.critics.base import infer_cond_rows  # noqa: E402
from impls.utils.datasets import make_dataset  # noqa: E402
from impls.utils.eval_cond import InferredCondAgent  # noqa: E402
from impls.utils.evaluation import evaluate  # noqa: E402
from impls.utils.flax_utils import resolve_run_dir, restore_agent  # noqa: E402
from env import get_adapter, make_config  # noqa: E402

FLAGS = flags.FLAGS
flags.DEFINE_string("critic", "fb", "Critic name.")
flags.DEFINE_string("actor", "ddpgbc", "Actor name.")
flags.DEFINE_string("domain", "cube_single", "Domain name.")
flags.DEFINE_multi_string("override", [], "Config override, dotlist form.")
flags.DEFINE_integer("seed", 0, "Seed (must match the checkpoint's training seed).")
flags.DEFINE_string("restore_path", "", "Glob matching exactly one run dir.")
flags.DEFINE_integer("restore_epoch", 500000, "params_{epoch}.pkl to restore.")
flags.DEFINE_integer("task_id", 1, "OGBench task_id. 1 == cube-single-play-singletask-v0.")
flags.DEFINE_integer("eval_episodes", 500, "Episodes. 500 is the only reportable count.")
flags.DEFINE_string("out", "", "Output JSON path.")
flags.DEFINE_bool("config_from_run", True, "Rebuild nets from the run's own config.json.")
flags.DEFINE_integer("infer_cond_episodes", 0, "See impls/main.py; 0 = goal-conditioned eval.")
flags.DEFINE_bool("eval_on_inferred_cond", False, "Evaluate on the inferred index instead "
                                                  "of the env's task goal.")
flags.DEFINE_integer("relabel_infer", 0,
                     "If >0, infer z from THIS MANY reward-relabelled transitions instead of "
                     "from the goal frame, and evaluate on it (impls/main.py's eval_relabel "
                     "path via OGBenchAdapter.relabel_infer_batch). This is the protocol that "
                     "matches PSMFlows -- psmflow/fb there get their task vector from a "
                     "reward-labelled batch, never from a goal observation. Needs the dataset "
                     "to carry 'qpos', so pass --override eval_relabel_size=<same N> too: that "
                     "key is what makes the adapter load the column.")
flags.DEFINE_float("relabel_shift", 1.0, "Reward shift for --relabel_infer (PSMFlows uses 1.0).")


def wilson(k, n, z=1.96):
    """Wilson score interval, identical to PSMFlows tools/eval_checkpoint.py::wilson."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - h) / d, 4), round((c + h) / d, 4)


def _recorded_config_overrides(run_dir):
    """Dotlist overrides reproducing the run's recorded config, minus eval-owned keys.
    Copied from scripts/reeval_checkpoint.py so the two cannot drift."""
    path = os.path.join(resolve_run_dir(run_dir), "config.json")
    if not os.path.isfile(path):
        print(f"[config_from_run] no config.json at {path}; using yaml defaults")
        return []
    with open(path) as f:
        recorded = json.load(f).get("config", {})
    skip = {"seed", "save_dir", "restore_path", "restore_epoch", "train_steps",
            "eval_episodes", "eval_interval", "save_interval", "log_interval",
            "exorl_task", "wandb", "run_group", "online"}
    out = []

    def walk(prefix, node):
        for k, v in node.items():
            if not prefix and k in skip:
                continue
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                walk(f"{key}.", v)
            elif isinstance(v, str):
                out.append(f'{key}="{v}"')
            else:
                out.append(f"{key}={json.dumps(v)}")

    walk("", recorded)
    return out


def main(_):
    base = (_recorded_config_overrides(FLAGS.restore_path)
            if FLAGS.config_from_run and FLAGS.restore_path else [])
    cfg = make_config(FLAGS.critic, FLAGS.actor, FLAGS.domain, base + list(FLAGS.override))
    np.random.seed(FLAGS.seed)

    adapter = get_adapter(cfg)
    env, train_ds, _val = adapter.make_env_and_datasets(cfg)
    train_ds = make_dataset(train_ds, cfg)
    ex = train_ds.sample(1)
    agent = Agent.create(FLAGS.critic, FLAGS.actor, FLAGS.seed,
                         jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]), cfg)
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch, params_only=True)

    eval_agent = agent
    if FLAGS.relabel_infer > 0:
        # Reward-inferred task vector, the PSMFlows-comparable route. The relabel is done by
        # the adapter against THIS task_id's goal_xyzs, never from the dataset's own rewards
        # column (cube play data is unlabelled and FBDataset zeroes it, which would make
        # FB's z = (r^T B)/N identically zero).
        nobs, rew = adapter.relabel_infer_batch(train_ds, env, FLAGS.task_id,
                                                FLAGS.relabel_infer,
                                                shift_reward=FLAGS.relabel_shift)
        eval_agent = InferredCondAgent(
            agent.infer_eval_cond(jnp.asarray(nobs), jnp.asarray(rew)))
        print(f"[relabel-infer] z from {FLAGS.relabel_infer} relabelled rows, "
              f"shift={FLAGS.relabel_shift}, task_id={FLAGS.task_id}", flush=True)
    elif FLAGS.infer_cond_episodes > 0:
        rb = train_ds.sample(FLAGS.infer_cond_episodes)
        rows = infer_cond_rows(agent.critic)
        eval_agent = agent.infer_eval_cond(jnp.asarray(rb[rows]), jnp.asarray(rb["rewards"]))
        if FLAGS.eval_on_inferred_cond:
            eval_agent = InferredCondAgent(eval_agent)

    info, trajs, _ = evaluate(agent=eval_agent, env=env, task_id=FLAGS.task_id, config=cfg,
                              num_eval_episodes=FLAGS.eval_episodes, num_video_episodes=0)

    # Per-episode successes off the trajectories, so the interval is computed on counts.
    # Same reducer as PSMFlows tools/eval_checkpoint.py.
    per_ep = [float(np.max(np.asarray(t["info"][-1].get("success", 0.0)))) for t in trajs]
    k, n = int(sum(p > 0.5 for p in per_ep)), len(per_ep)
    lo, hi = wilson(k, n)

    report = {
        "env": f"{cfg['env_name']} task_id={FLAGS.task_id}",
        "agent": f"factored-fb {FLAGS.critic} x {FLAGS.actor}",
        "acting": (f"reward-inferred z (relabel N={FLAGS.relabel_infer}, "
                   f"shift={FLAGS.relabel_shift})"
                   if FLAGS.relabel_infer > 0 else
                   ("inferred z" if FLAGS.eval_on_inferred_cond else
                    "goal-conditioned z = project_z(B(goal))")),
        "acting_mode": (f"critic={FLAGS.critic} actor={FLAGS.actor} "
                        f"actor.alpha={cfg['actor'].get('alpha')} "
                        f"ortho_coef={cfg.get('ortho_coef')} z_dim={cfg.get('z_dim')}"),
        "bc_alpha": cfg["actor"].get("alpha"),
        "ortho_coef": cfg.get("ortho_coef"),
        "task_id": FLAGS.task_id,
        "restore_path": str(FLAGS.restore_path),
        "restore_epoch": int(FLAGS.restore_epoch),
        "seed": int(FLAGS.seed),
        "num_episodes": n,
        "num_success": k,
        "success": round(k / n, 4) if n else None,
        "wilson95": [lo, hi],
        "half_width": round((hi - lo) / 2, 4),
        "eval_success_field": round(float(info["success"]), 4),
        "per_episode_success": [int(p > 0.5) for p in per_ep],
    }
    print(f"\nfb-eval500 [{report['env']}] {k}/{n} = {report['success']}  "
          f"95% CI [{lo}, {hi}]", flush=True)
    if FLAGS.out:
        os.makedirs(os.path.dirname(os.path.abspath(FLAGS.out)), exist_ok=True)
        with open(FLAGS.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"wrote {FLAGS.out}")


if __name__ == "__main__":
    app.run(main)
