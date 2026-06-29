"""scripts/data/extract_task_anchors.py — per-task EEF / cube / goal anchors.

For each cube-single-play task (1..5), reset the OGBench env, query the
authoritative starting cube xy + goal xy from `cur_task_info`, and read
the home end-effector xy from the proprio info. Cached to
analysis/misc/scene/task_anchors.json so figure renderers can plot the three
positions without needing MuJoCo themselves.

Run in any venv that has the vendored OGBench cube env (.venv works).
MUJOCO_GL=glfw on macOS (only needed for an offscreen renderer; reset()
itself doesn't render but the env may import the renderer module).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _eef_xy(env, info) -> tuple[float, float]:
    """EEF xy at the home pose. Prefer proprio info from reset; fall
    back to a zero-action step; fall back to the EEF site xpos."""
    key = "proprio/effector_pos"
    if isinstance(info, dict) and key in info:
        p = np.asarray(info[key], float).reshape(-1)
        return float(p[0]), float(p[1])
    try:
        a = np.zeros_like(env.action_space.sample(), dtype=np.float32)
        _, _, _, _, info2 = env.step(a)
        if key in info2:
            p = np.asarray(info2[key], float).reshape(-1)
            return float(p[0]), float(p[1])
    except Exception:
        pass
    # final fallback: read EEF-ish site directly
    import mujoco
    u = env.unwrapped
    for nm in ("attachment_site", "tcp", "pinch", "ee", "robotiq/eef"):
        sid = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_SITE, nm)
        if sid >= 0:
            p = np.asarray(u.data.site_xpos[sid], float)
            return float(p[0]), float(p[1])
    raise RuntimeError("could not determine EEF xy from env")


def extract(env_name: str = "cube-single-play-v0",
            task_ids=(1, 2, 3, 4, 5)) -> dict:
    ogb = REPO_ROOT / "third_party" / "ogbench" / "impls"
    if str(ogb) not in sys.path:
        sys.path.insert(0, str(ogb))
    import ogbench

    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    out: dict = {}
    try:
        for t in task_ids:
            _, info = env.reset(options=dict(task_id=int(t)))
            u = env.unwrapped
            tb = int(getattr(u, "_target_block", 0) or 0)
            cube0 = np.asarray(u.cur_task_info["init_xyzs"][tb], float)
            goal = np.asarray(u.cur_task_info["goal_xyzs"][tb], float)
            eef = _eef_xy(env, info)
            out[f"task{t}"] = {
                "cube_init": [float(cube0[0]), float(cube0[1])],
                "goal": [float(goal[0]), float(goal[1])],
                "eef_init": [eef[0], eef[1]],
            }
    finally:
        env.close()
    return out


def main() -> None:
    anchors = extract()
    out = REPO_ROOT / "analysis" / "misc" / "scene" / "task_anchors.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(anchors, indent=2))
    print(f"[task_anchors] wrote {len(anchors)} entries -> {out}")
    for k, v in anchors.items():
        print(f"  {k}: cube={v['cube_init']} goal={v['goal']} "
              f"eef={v['eef_init']}")


if __name__ == "__main__":
    main()
