"""scripts/figures/fb_pixel_curves.py — FB (and optionally GCIQL) pixel eval curves.

Loads FB pixel run histories from a wandb_pull cache and produces per-task +
aggregate IQM success curves (95% stratified-bootstrap CI) and a final IQM
success table, reusing the method-generic machinery in fb_gciql_curves.

GCIQL pixel runs are not available yet; pass --gciql <eval.csv root> to add
them when they exist (the curves + table then include both methods).

Usage:
    python scripts/figures/fb_pixel_curves.py \
        --fb analysis/wandb/wandb_data_fbpixel \
        --out analysis/curves/fb_pixel
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures.fb_gciql_curves import COMMON_GRID, load_fb, load_gciql, render


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fb", default=str(REPO_ROOT / "analysis"
                                        / "wandb" / "wandb_data_fbpixel"))
    ap.add_argument("--gciql", default=None,
                    help="optional GCIQL pixel eval.csv root")
    ap.add_argument("--max-step", type=int, default=None,
                    help="cap the shared step grid for a matched-budget "
                         "comparison (e.g. 500000 when GCIQL trains to 500k); "
                         "the final table is then taken at this step too")
    ap.add_argument("--out", default=str(REPO_ROOT / "analysis" / "curves" / "fb_pixel"))
    args = ap.parse_args()

    methods = {"FB": load_fb(args.fb)}
    if args.gciql:
        methods["GCIQL"] = load_gciql(args.gciql)

    if args.max_step is not None:
        grid = np.arange(0, args.max_step + 1, 100_000)
        final_at_grid_end = True
        title = f"FB vs GCIQL (pixels, matched {args.max_step // 1000}k)"
    else:
        grid = COMMON_GRID
        final_at_grid_end = False
        title = "FB (pixels)"

    print(f"[fb_pixel_curves] methods={list(methods)} "
          f"FB seeds={sorted(methods['FB'])} max_step={args.max_step}")
    render(methods, Path(args.out), title_prefix=title,
           n_seeds_note="n=5 seeds", grid=grid,
           final_at_grid_end=final_at_grid_end)


if __name__ == "__main__":
    main()
