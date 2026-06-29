"""scripts/figures/outcome_clips.py — one success + one failure rollout per task.

Rolls out a checkpoint with frame rendering, picks one successful and one
failed episode per cube-single task, and saves them both as GIFs and as a
2-row (success / failure) static filmstrip for the paper. Two backends:

  --method fb    : torch FB checkpoint (run under .venv)
  --method gciql : JAX GCIQL checkpoint (run under .venv-jax-cpu)

The pure helpers (outcome selection, filmstrip stacking) are JAX/torch-free
and unit-tested.

Usage (FB):
    .venv/bin/python scripts/figures/outcome_clips.py --method fb \
        --config <run>/.hydra/config.yaml \
        --checkpoint <run>/checkpoints/<ckpt>.pt \
        --data-path <dataset> --mujoco-gl glfw --tasks 1,2,3,4,5 \
        --n-episodes 40 --tag fb_state \
        --out RESULTS/outcome_clips/fb_state \
        --filmstrip-out PAPER/fb-cube-failure/figures

Usage (GCIQL):
    .venv-jax-cpu/bin/python scripts/figures/outcome_clips.py --method gciql \
        --run-dir <sd...> --step 500000 --obs-type pixels \
        --tasks 1,2,3,4,5 --n-episodes 40 --tag gciql_pixel \
        --out RESULTS/outcome_clips/gciql_pixel \
        --filmstrip-out PAPER/fb-cube-failure/figures --mujoco-gl glfw
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures.pixel_filmstrips import make_filmstrip, sample_frames

TASK_TMPL = "cube-single-play-singletask-task{n}-v0"


def pick_success_failure(successes: List[bool], frames: List[List[np.ndarray]]
                         ) -> Tuple[Optional[List[np.ndarray]],
                                    Optional[List[np.ndarray]]]:
    """First successful and first failed episode's frames (each or None)."""
    succ = next((frames[i] for i, s in enumerate(successes) if s), None)
    fail = next((frames[i] for i, s in enumerate(successes) if not s), None)
    return succ, fail


def two_row_filmstrip(succ_frames: Optional[List[np.ndarray]],
                      fail_frames: Optional[List[np.ndarray]],
                      n_frames: int = 6, pad: int = 2, gap: int = 8):
    """Stack a success row over a failure row (each a sampled filmstrip).
    Missing rows are skipped; returns None if both are missing."""
    from PIL import Image

    rows = []
    for fr in (succ_frames, fail_frames):
        if fr:
            rows.append(make_filmstrip(sample_frames(fr, n_frames), pad=pad))
    if not rows:
        return None
    w = max(r.width for r in rows)
    h = sum(r.height for r in rows) + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (w, h), "white")
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + gap
    return canvas


def compose_with_goal(goal_frame: np.ndarray, strip, sep: int = 10):
    """Prepend a goal-reference panel (left) to a filmstrip, resized to the
    strip's height (preserving aspect), with a white separator."""
    from PIL import Image

    g = Image.fromarray(np.asarray(goal_frame)).convert("RGB")
    h = strip.height
    w = max(1, round(g.width * h / g.height))
    g = g.resize((w, h))
    canvas = Image.new("RGB", (g.width + sep + strip.width, h), "white")
    canvas.paste(g, (0, 0))
    canvas.paste(strip, (g.width + sep, 0))
    return canvas


def frame_with_goal(goal_frame: np.ndarray, frame: np.ndarray,
                    sep: int = 6) -> np.ndarray:
    """One animated frame with the static goal panel on its left (goal resized
    to the frame height). Used to bake the goal into the GIFs."""
    from PIL import Image

    fr = Image.fromarray(np.asarray(frame)).convert("RGB")
    g = Image.fromarray(np.asarray(goal_frame)).convert("RGB")
    h = fr.height
    w = max(1, round(g.width * h / g.height))
    g = g.resize((w, h))
    canvas = Image.new("RGB", (g.width + sep + fr.width, h), "white")
    canvas.paste(g, (0, 0))
    canvas.paste(fr, (g.width + sep, 0))
    return np.asarray(canvas)


def render_goal_frame(env) -> Optional[np.ndarray]:
    """Render the scene with the cube placed at the task goal. The rollout
    render shows only the live scene (OGBench conveys the goal as a separate
    signal), so we set the cube free-joint to ``cur_task_info['goal_xyzs']``
    and render. Returns None if the env doesn't expose the manip internals."""
    try:
        import mujoco
        u = env.unwrapped
        tb = int(getattr(u, "_target_block", 0) or 0)
        goal = np.asarray(u.cur_task_info["goal_xyzs"][tb], np.float64)
        u._data.joint("object_joint_0").qpos[:3] = goal
        mujoco.mj_forward(u._model, u._data)
        return np.asarray(env.render())
    except Exception as e:  # noqa: BLE001 - goal panel is best-effort
        print(f"  [warn] render_goal_frame failed: {e}")
        return None


def _save_gif(frames: List[np.ndarray], path: Path, fps: int = 30) -> None:
    import imageio
    imageio.mimsave(str(path), [np.asarray(f) for f in frames],
                    format="GIF", fps=fps, loop=0)


def _emit(tag: str, task_n: int, succ, fail, out_dir: Path,
          film_dir: Path, n_frames: int, goal_frame=None) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    film_dir.mkdir(parents=True, exist_ok=True)
    if succ:
        _save_gif(succ, out_dir / f"task{task_n}_success.gif")
    if fail:
        _save_gif(fail, out_dir / f"task{task_n}_failure.gif")
    strip = two_row_filmstrip(succ, fail, n_frames=n_frames)
    if strip is not None:
        if goal_frame is not None:
            strip = compose_with_goal(goal_frame, strip)
        strip.save(film_dir / f"clip_{tag}_task{task_n}.png")
    status = ("ok" if (succ and fail)
              else "MISSING failure" if succ else "MISSING success"
              if fail else "MISSING both")
    print(f"[outcome_clips] {tag} task{task_n}: {status}")
    return status


# ---------------------------------------------------------------------------
# FB backend (torch / .venv)
# ---------------------------------------------------------------------------

def _fb_clips(args) -> None:
    import torch  # noqa: F401
    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from envs.ogbench import create_ogbench_env
    from envs.rollout import rollout
    from evals.ogbench import OGBenchEvaluator
    from data.ogbench import load_ogbench_dataset

    cfg = load_cfg(args.config, device=args.device)
    if args.data_path:
        cfg.data_path = args.data_path
    fs = int(getattr(cfg, "frame_stack", 1))
    env, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint, map_location=args.device)
    if hasattr(env, "close"):
        env.close()

    buffer = load_ogbench_dataset(
        domain=cfg.domain, data_path=cfg.data_path,
        load_n_episodes=cfg.load_n_episodes, device=args.device,
        n_transitions=cfg.n_transitions, obs_type=cfg.obs_type, frame_stack=fs)
    evaluator = OGBenchEvaluator(
        domain=cfg.domain, agent=agent, offline_buffer=buffer,
        relabel_size=cfg.eval_relabel_size, n_episodes=args.n_episodes,
        shift_reward=cfg.eval_shift_reward, obs_type=cfg.obs_type,
        frame_stack=fs, seed=cfg.seed, device=args.device, use_wandb=False)

    out_dir, film_dir = Path(args.out), Path(args.filmstrip_out)
    for n in [int(t) for t in args.tasks.split(",") if t.strip()]:
        task = TASK_TMPL.format(n=n)
        z, _ = evaluator._infer_z(task)
        e, _ = create_ogbench_env(task, seed=cfg.seed, obs_type=cfg.obs_type,
                                  frame_stack=fs)
        try:
            gf = render_goal_frame(e)
            _, infos, frames = rollout(e, agent, args.n_episodes, ctx=z,
                                       record=True)
        finally:
            e.close()
        successes = [any(si.get("success", False) for si in ep) for ep in infos]
        succ, fail = pick_success_failure(successes, frames)
        _emit(args.tag, n, succ, fail, out_dir, film_dir, args.frames,
              goal_frame=gf)


# ---------------------------------------------------------------------------
# GCIQL backend (JAX / .venv-jax-cpu)
# ---------------------------------------------------------------------------

def _gciql_clips(args) -> None:
    ogb_impls = REPO_ROOT / "third_party" / "ogbench" / "impls"
    if str(ogb_impls) not in sys.path:
        sys.path.insert(0, str(ogb_impls))
    import ogbench  # vendored
    import jax
    from agents.gciql import GCIQLAgent, get_config
    from utils.flax_utils import restore_agent
    from scripts.profiles.gciql_profile import parse_flags

    flags = parse_flags(args.run_dir)
    config = get_config()
    saved = flags.get("agent")
    if isinstance(saved, dict):  # overlay trained agent config (e.g. encoder)
        for k, v in saved.items():
            if k in config:
                config[k] = v

    out_dir, film_dir = Path(args.out), Path(args.filmstrip_out)
    for n in [int(t) for t in args.tasks.split(",") if t.strip()]:
        env = ogbench.make_env_and_datasets(flags["env_name"], env_only=True)
        ex_obs, info = env.reset(options=dict(task_id=n))
        agent = GCIQLAgent.create(
            flags["seed"], np.asarray(ex_obs, np.float32)[None],
            np.asarray(env.action_space.sample(), np.float32)[None], config)
        agent = restore_agent(agent, str(args.run_dir), args.step)
        gf = render_goal_frame(env)
        rng = jax.random.PRNGKey(0)
        successes: List[bool] = []
        frames: List[List[np.ndarray]] = []
        for _ in range(args.n_episodes):
            obs, info = env.reset(options=dict(task_id=n))
            goal = info["goal"]
            ep_frames = [np.asarray(env.render())]
            succ = False
            for _ in range(args.max_steps):
                rng, key = jax.random.split(rng)
                a = np.asarray(agent.sample_actions(
                    observations=np.asarray(obs, np.float32),
                    goals=np.asarray(goal, np.float32),
                    seed=key, temperature=0.0))
                a = np.clip(a, -1.0, 1.0)
                obs, _, term, trunc, info = env.step(a)
                ep_frames.append(np.asarray(env.render()))
                succ = succ or bool(info.get("success", False))
                if term or trunc:
                    break
            successes.append(succ)
            frames.append(ep_frames)
        env.close()
        succ_f, fail_f = pick_success_failure(successes, frames)
        _emit(args.tag, n, succ_f, fail_f, out_dir, film_dir, args.frames,
              goal_frame=gf)


def _rebuild_filmstrips(args) -> None:
    """Rebuild filmstrips (with a goal-reference panel) from already-saved
    GIFs, without re-running rollouts. The goal scene is method/regime
    independent, so we render it once per task from the state env."""
    import imageio.v2 as imageio
    from envs.ogbench import create_ogbench_env

    tasks = [int(t) for t in args.tasks.split(",") if t.strip()]
    film_dir = Path(args.filmstrip_out)
    film_dir.mkdir(parents=True, exist_ok=True)
    goals: Dict[int, Optional[np.ndarray]] = {}
    for n in tasks:
        e, _ = create_ogbench_env(TASK_TMPL.format(n=n), seed=0,
                                  obs_type="state", render_height=240,
                                  render_width=240)
        try:
            goals[n] = render_goal_frame(e)
        finally:
            e.close()
    for tag in [t for t in args.tags.split(",") if t.strip()]:
        gif_dir = Path(args.gif_root) / tag
        for n in tasks:
            sp = gif_dir / f"task{n}_success.gif"
            fp = gif_dir / f"task{n}_failure.gif"
            succ = list(imageio.mimread(str(sp))) if sp.exists() else None
            fail = list(imageio.mimread(str(fp))) if fp.exists() else None
            strip = two_row_filmstrip(succ, fail, n_frames=args.frames)
            if strip is None:
                print(f"  [warn] {tag} task{n}: no gifs in {gif_dir}")
                continue
            gf = goals.get(n)
            if gf is not None:
                strip = compose_with_goal(gf, strip)
            strip.save(film_dir / f"clip_{tag}_task{n}.png")
            # Goal-paneled GIFs (non-destructive: separate *_goal.gif).
            if gf is not None:
                for outcome, frames in (("success", succ), ("failure", fail)):
                    if frames:
                        _save_gif([frame_with_goal(gf, f) for f in frames],
                                  gif_dir / f"task{n}_{outcome}_goal.gif")
            print(f"[outcome_clips] rebuilt clip_{tag}_task{n}.png (+goal gifs)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True,
                    choices=["fb", "gciql", "filmstrips"])
    ap.add_argument("--tasks", default="1,2,3,4,5")
    ap.add_argument("--n-episodes", type=int, default=40)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--tag", help="output label, e.g. fb_pixel")
    ap.add_argument("--out", help="dir for GIFs")
    ap.add_argument("--filmstrip-out", required=True,
                    help="dir for the 2-row filmstrip PNGs")
    ap.add_argument("--gif-root", help="(filmstrips mode) parent of <tag> "
                    "GIF dirs to rebuild from")
    ap.add_argument("--tags", help="(filmstrips mode) comma-separated tags")
    ap.add_argument("--mujoco-gl", default=None)
    # FB
    ap.add_argument("--config")
    ap.add_argument("--checkpoint")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-path")
    # GCIQL
    ap.add_argument("--run-dir")
    ap.add_argument("--step", type=int, default=1000000)
    ap.add_argument("--obs-type", default=None)
    ap.add_argument("--max-steps", type=int, default=1000)
    args = ap.parse_args()

    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    if args.method == "filmstrips":
        assert args.gif_root and args.tags, "filmstrips needs --gif-root/--tags"
        _rebuild_filmstrips(args)
    elif args.method == "fb":
        assert args.config and args.checkpoint, "fb needs --config/--checkpoint"
        assert args.tag and args.out, "fb needs --tag/--out"
        _fb_clips(args)
    else:
        assert args.run_dir, "gciql needs --run-dir"
        assert args.tag and args.out, "gciql needs --tag/--out"
        _gciql_clips(args)


if __name__ == "__main__":
    main()
