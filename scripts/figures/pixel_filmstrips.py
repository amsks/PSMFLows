"""scripts/figures/pixel_filmstrips.py — static filmstrips from eval GIFs.

For each cube-single task, sample evenly-spaced frames from a representative
seed's eval GIF at a chosen checkpoint and composite a horizontal filmstrip PNG
for the paper (a PDF cannot embed animation). Works for any run that saved
rendered eval GIFs (pixel or state runs, since the env renders regardless of the
policy's observation type); `--prefix` sets the output filename prefix.

Usage:
    python scripts/figures/pixel_filmstrips.py \
        --results-root RESULTS/fb-pixel-results \
        --seed 0 --step step_1000000 --frames 6 --prefix pixel \
        --out PAPER/fb-cube-failure/figures
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

TASKS = [1, 2, 3, 4, 5]


def sample_frames(frames: List[np.ndarray], n: int) -> List[np.ndarray]:
    """n evenly-spaced frames, inclusive of the first and last."""
    if len(frames) <= n:
        return list(frames)
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


def make_filmstrip(frames: List[np.ndarray], pad: int = 2) -> Image.Image:
    """Composite frames into one horizontal strip with white separators."""
    imgs = [Image.fromarray(np.asarray(f)).convert("RGB") for f in frames]
    h = max(im.height for im in imgs)
    w = sum(im.width for im in imgs) + pad * (len(imgs) - 1)
    strip = Image.new("RGB", (w, h), "white")
    x = 0
    for im in imgs:
        strip.paste(im, (x, 0))
        x += im.width + pad
    return strip


def find_run_dir(results_root: Path, seed: int) -> Path:
    matches = sorted(glob.glob(str(results_root / f"*__s{seed}")))
    if not matches:
        raise FileNotFoundError(
            f"no run dir matching *__s{seed} under {results_root}")
    return Path(matches[0])


def build(results_root: Path, seed: int, step: str, n_frames: int,
          out_dir: Path, prefix: str = "pixel") -> List[Path]:
    import imageio.v2 as imageio

    run = find_run_dir(results_root, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for t in TASKS:
        gif = (run / "eval_videos" / step
               / f"cube-single-play-singletask-task{t}-v0.gif")
        frames = imageio.mimread(str(gif))
        strip = make_filmstrip(sample_frames(frames, n_frames))
        dest = out_dir / f"{prefix}_filmstrip_task{t}.png"
        strip.save(dest)
        written.append(dest)
        print(f"[pixel_filmstrips] wrote {dest} ({len(frames)} src frames)")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="RESULTS/fb-pixel-results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", default="step_1000000")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--prefix", default="pixel",
                    help="output filename prefix (e.g. pixel, state)")
    ap.add_argument("--out", default="PAPER/fb-cube-failure/figures")
    args = ap.parse_args()
    build(Path(args.results_root), args.seed, args.step, args.frames,
          Path(args.out), prefix=args.prefix)


if __name__ == "__main__":
    main()
