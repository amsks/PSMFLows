"""evals/scene_calib.py — fixed analytic world(x,y)->pixel mapping for
the top-down cube scene. We DEFINE this affine (workspace box ->
inset pixel box) and aim the MuJoCo camera to match it; this keeps the
overlay alignment exact and render-independent. torch/jax/mujoco-free."""

from __future__ import annotations

from typing import Dict


def make_calib(xmin: float, xmax: float, ymin: float, ymax: float,
                img_w: int, img_h: int, margin: int = 50) -> Dict:
    """Workspace [xmin,xmax]x[ymin,ymax] (metres) -> pixel box inset by
    `margin`. World +x -> +px (right); world +y -> -py (up), since
    image y grows downward."""
    ax = (img_w - 2 * margin) / (xmax - xmin)
    bx = margin - ax * xmin
    ay = -(img_h - 2 * margin) / (ymax - ymin)
    by = (img_h - margin) - ay * ymin
    return {"ax": ax, "bx": bx, "ay": ay, "by": by,
            "img_w": int(img_w), "img_h": int(img_h),
            "workspace": {"xmin": xmin, "xmax": xmax,
                          "ymin": ymin, "ymax": ymax},
            "margin": int(margin)}


def world_to_px(world_x: float, world_y: float, calib: Dict):
    """(x,y) metres -> (px,py) float pixels under `calib`."""
    px = calib["ax"] * world_x + calib["bx"]
    py = calib["ay"] * world_y + calib["by"]
    return px, py
