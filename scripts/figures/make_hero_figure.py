#!/usr/bin/env python
"""scripts/figures/make_hero_figure.py — lead figure: the value can't tell two grasps apart.

Finds two in-hand grasps that the method scores almost equally (same value) but
whose on-policy rollouts diverge---one delivers the cube to the goal, the other
drops it. Renders both the grasp and its outcome, robotics-filmstrip style, to
make the interaction-gap aliasing physical. Run under .venv. MUJOCO_GL=glfw.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--method", default="FB")
    ap.add_argument("--task-id", type=int, default=2)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--max-steps", type=int, default=200)
    args = ap.parse_args()
    os.environ["MUJOCO_GL"] = "glfw"

    import mujoco
    import torch
    import matplotlib.pyplot as plt
    from PIL import Image
    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from evals.ogbench import OGBenchEvaluator
    from envs.ogbench import ALL_TASKS
    from data.ogbench import load_ogbench_dataset
    from scripts.value.counterfactual_value_probe import sample_grasps, value_q, _zb

    cfg = load_cfg(args.config, device="cpu"); cfg.data_path = "datasets"
    env, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint)
    if hasattr(env, "close"):
        env.close()
    model = agent.model
    buf = load_ogbench_dataset(domain=cfg.domain, data_path=cfg.data_path,
                               load_n_episodes=cfg.load_n_episodes, device=cfg.device,
                               n_transitions=cfg.n_transitions, obs_type=cfg.obs_type)
    ev = OGBenchEvaluator(domain=cfg.domain, agent=agent, offline_buffer=buf,
                          relabel_size=cfg.eval_relabel_size, n_episodes=1,
                          shift_reward=cfg.eval_shift_reward, obs_type=cfg.obs_type,
                          seed=cfg.seed, device=cfg.device, use_wandb=False)
    tasks = list(ALL_TASKS.get(cfg.domain, []))
    z = ev._infer_z(tasks[args.task_id - 1])[0]

    obs_g, phys_g = sample_grasps("datasets", cfg.domain, args.n)
    val = value_q(model, agent, obs_g, z)

    import ogbench
    e = ogbench.make_env_and_datasets(cfg.domain, env_only=True, width=640, height=480)
    u = e.unwrapped; nv = u._model.nv

    CROP = (slice(95, 470), slice(60, 600))   # drop empty starfield

    def rollout(qpos):
        e.reset(options=dict(task_id=args.task_id))
        u.set_state(np.asarray(qpos, np.float64), np.zeros(nv))
        u._data.ctrl[u._gripper_actuator_ids] = 0.0
        mujoco.mj_forward(u._model, u._data)
        grasp_img = np.asarray(u.render(camera="front"))[CROP]
        ob = np.asarray(u.compute_observation(), np.float32)
        s = False
        out_img = None
        for _ in range(args.max_steps):
            a = np.clip(np.asarray(agent.act(torch.as_tensor(ob[None]), _zb(z, 1))).reshape(-1), -1, 1)
            ob_, _, term, trunc, info = e.step(a); ob = np.asarray(ob_, np.float32)
            if bool(info.get("success", False)) and not s:   # capture delivery moment
                s = True
                out_img = np.asarray(u.render(camera="front"))[CROP]
                break
            if term or trunc:
                break
        if out_img is None:                               # fail: final (dropped) frame
            out_img = np.asarray(u.render(camera="front"))[CROP]
        return s, grasp_img, out_img

    succ = np.array([rollout(p)[0] for p in phys_g])
    print(f"task {args.task_id}: success rate {succ.mean():.2f} ({succ.sum()}/{len(succ)})")

    # matched pair: success vs fail grasp that is both value-close AND
    # observation-near (so the grasps look alike and the value agrees).
    pos = np.where(succ)[0]; neg = np.where(~succ)[0]
    vstd = val.std() + 1e-8
    ostd = obs_g.std(0) + 1e-8
    best = None
    for i in pos:
        for j in neg:
            dv = abs(val[i] - val[j]) / vstd
            do = np.linalg.norm((obs_g[i] - obs_g[j]) / ostd) / np.sqrt(obs_g.shape[1])
            score = dv + do
            if best is None or score < best[0]:
                best = (score, i, j)
    _, i, j = best
    print(f"pair: success idx {i} (V={val[i]:.2f}) vs fail idx {j} (V={val[j]:.2f}); "
          f"|dV|={abs(val[i]-val[j]):.3f}")

    _, g_i, f_i = rollout(phys_g[i])
    _, g_j, f_j = rollout(phys_g[j])

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2))
    GREEN, RED = "#2ca02c", "#d62728"
    panels = [(axes[0, 0], g_i, f"grasp A   value $V={val[i]:.2f}$", "black"),
              (axes[0, 1], f_i, "outcome A:  cube delivered ✓", GREEN),
              (axes[1, 0], g_j, f"grasp B   value $V={val[j]:.2f}$", "black"),
              (axes[1, 1], f_j, "outcome B:  cube dropped ✗", RED)]
    for ax, img, title, col in panels:
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=12, fontweight="bold", color=col)
    for r in (0, 1):
        axes[r, 0].annotate("", xy=(-0.02, 0.5), xytext=(-0.14, 0.5),
                            xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="->", lw=2, color="0.4"))
    fig.suptitle(f"Two grasps {args.method} values almost identically "
                 f"($V{{=}}{val[i]:.2f}$ vs ${val[j]:.2f}$) --- yet one is delivered and one is dropped.\n"
                 "The value the policy follows cannot separate a controllable grasp from a failing one.",
                 fontsize=12.5, y=1.0)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    out = REPO / "PAPER" / "rlbrew" / "figures" / "fig_hero.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    Image.fromarray(g_i).save("/tmp/hero_grasp_A.png")
    print(f"[hero] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
