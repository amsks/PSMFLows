import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.scene_calib import make_calib, world_to_px


def test_calib_maps_workspace_corners_to_pixel_box():
    c = make_calib(xmin=0.30, xmax=0.55, ymin=-0.25, ymax=0.25,
                    img_w=600, img_h=600, margin=50)
    # x-min/x-max -> horizontal pixel box [margin, img_w-margin]
    px0, _ = world_to_px(0.30, 0.0, c)
    px1, _ = world_to_px(0.55, 0.0, c)
    assert round(px0) == 50 and round(px1) == 550
    # world y maps into the vertical box, inverted (image y grows down)
    _, py0 = world_to_px(0.40, -0.25, c)
    _, py1 = world_to_px(0.40, 0.25, c)
    assert round(py0) == 550 and round(py1) == 50
    # center maps to image center
    pcx, pcy = world_to_px(0.425, 0.0, c)
    assert round(pcx) == 300 and round(pcy) == 300
