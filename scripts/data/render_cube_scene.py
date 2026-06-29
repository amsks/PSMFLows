"""scripts/data/render_cube_scene.py — one top-down background of the cube
scene + calib.json. Tries an offscreen MuJoCo render; on failure
(macOS offscreen GL) writes a schematic top-down canvas instead. The
world->pixel affine is fixed by evals.scene_calib (camera is aimed to
match it), so downstream overlays align either way.

Run in .venv-jax-cpu (vendored OGBench cube env has render())."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evals.scene_calib import make_calib, world_to_px  # noqa: E402

# Workspace widened beyond the calibration box so the photoreal
# top-down covers the actual cube-xy data extent
# (x in [0.245, 0.606], y in [-0.35, 0.32]) with margin.
XMIN, XMAX, YMIN, YMAX = 0.20, 0.65, -0.40, 0.40
# Default mujoco model offscreen buffer is ~640x480; stay within it.
IMG_W = IMG_H = 480
MARGIN = 40


def _try_mujoco_topdown(out_png: Path) -> bool:
    """Render a top-down RGB frame via the vendored OGBench cube env.
    Returns True on success, False if offscreen GL is unavailable."""
    try:
        ogb = REPO_ROOT / "third_party" / "ogbench" / "impls"
        if str(ogb) not in sys.path:
            sys.path.insert(0, str(ogb))
        import ogbench  # noqa: F401
        import mujoco

        env = ogbench.make_env_and_datasets("cube-single-play-v0",
                                             env_only=True)
        env.reset(options=dict(task_id=1))
        m = env.unwrapped.model
        d = env.unwrapped.data
        # The cube model's default offscreen framebuffer can be smaller
        # than our requested image; enlarge it before the Renderer.
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), IMG_W)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), IMG_H)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.5 * (XMIN + XMAX), 0.5 * (YMIN + YMAX), 0.05]
        cam.elevation = -89.9
        cam.azimuth = 90.0
        cam.distance = 1.2
        # Robot is in geom groups 2-3; world (table) + cube + goal
        # marker are all in group 1. Show only group 1 (drop the robot)
        # and make the task-specific cube/goal geoms invisible so the
        # background stays task-agnostic.
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        opt.geomgroup[1] = 1
        for nm in ("object_0", "target_object_0"):
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, nm)
            if gid >= 0:
                m.geom_rgba[gid, 3] = 0.0
        ren = mujoco.Renderer(m, IMG_H, IMG_W)
        ren.update_scene(d, camera=cam, scene_option=opt)
        img = ren.render()
        ren.close()
        env.close()
        plt.imsave(str(out_png), np.asarray(img))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [render_cube_scene] mujoco offscreen failed: {e}")
        return False


def _schematic_topdown(out_png: Path) -> None:
    """Fallback: schematic top-down table-plane canvas (clearly not
    photoreal). Table workspace rectangle + label."""
    fig = plt.figure(figsize=(IMG_W / 100, IMG_H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((MARGIN, MARGIN),
                               IMG_W - 2 * MARGIN, IMG_H - 2 * MARGIN,
                               facecolor="#d9d2c5", edgecolor="#555"))
    ax.text(IMG_W / 2, 24, "SCHEMATIC (offscreen render unavailable)",
            ha="center", fontsize=8, color="#555")
    fig.savefig(str(out_png))
    plt.close(fig)


def main() -> None:
    out = REPO_ROOT / "analysis" / "misc" / "scene"
    out.mkdir(parents=True, exist_ok=True)
    png = out / "topdown.png"
    photoreal = _try_mujoco_topdown(png)
    if not photoreal:
        _schematic_topdown(png)
    calib = make_calib(XMIN, XMAX, YMIN, YMAX, IMG_W, IMG_H, MARGIN)
    calib["photoreal"] = bool(photoreal)
    (out / "calib.json").write_text(json.dumps(calib, indent=2))

    # smoke validation: known task1 init/goal must land in-image
    init_xy = (0.425, 0.10)
    goal_xy = (0.425, -0.10)
    for name, (wx, wy) in (("init", init_xy), ("goal", goal_xy)):
        px, py = world_to_px(wx, wy, calib)
        ok = 0 <= px <= IMG_W and 0 <= py <= IMG_H
        print(f"  [render_cube_scene] {name} ({wx},{wy}) -> "
              f"({px:.1f},{py:.1f}) in-image={ok}")
    print(f"[render_cube_scene] photoreal={photoreal} -> {out}")


if __name__ == "__main__":
    main()
