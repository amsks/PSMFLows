#!/usr/bin/env bash
# Local smoke for FB pixel representation_profile. Builds a tiny SYNTHETIC
# pixel buffer (real physics from the local state dataset + placeholder pixel
# frames) and runs a reduced profile against the real s0 pixel checkpoint.
# Verifies plumbing + cube-space coverage; not scientific validity.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PY:-.venv/bin/python}"
STATE_DS="${STATE_DS:-datasets/cube-single-play-v0/buffer}"
MUJOCO_GL="${MUJOCO_GL:-glfw}"

S0_DIR=$(ls -d RESULTS/fb-pixel-results/*__s0 2>/dev/null | head -1)
[ -n "$S0_DIR" ] || { echo "FAIL: no s0 pixel run dir under RESULTS/fb-pixel-results"; exit 1; }
[ -d "$STATE_DS" ] || { echo "FAIL: no local state dataset at $STATE_DS"; exit 1; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
SYN="$TMP/cube-single-play-v0/buffer"
mkdir -p "$SYN"

"$PY" - "$STATE_DS" "$SYN" <<'PY'
import sys
import numpy as np
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(src.glob("episode_*.npz"))[:8]
assert files, f"no state episodes under {src}"
for f in files:
    z = dict(np.load(f))
    T = z["observation"].shape[0]
    z["pixels"] = np.zeros((T, 64, 64, 3), np.uint8)
    np.savez(dst / f.name, **z)
print(f"[smoke] wrote {len(files)} synthetic pixel episodes -> {dst}")
PY

OUT="$TMP/repr/s0_final"
"$PY" scripts/probes/representation_profile.py \
    --config "$S0_DIR/.hydra/config.yaml" \
    --checkpoint "$S0_DIR/checkpoints/final.pt" \
    --out "$OUT" --data-path "$TMP" --mujoco-gl "$MUJOCO_GL" \
    --tasks cube-single-play-singletask-task1-v0 \
    --n-episodes 2 --buffer-sample 256

"$PY" - "$OUT" <<'PY'
import sys
import pandas as pd
from pathlib import Path
out = Path(sys.argv[1])
for name in ("value_landscape", "value_steps", "z_decoding",
             "b_resolution", "coverage"):
    p = out / f"{name}.parquet"
    assert p.exists(), f"missing {p}"
cov = pd.read_parquet(out / "coverage.parquet")
assert {"outcome", "region", "nn_dist"} <= set(cov.columns), list(cov.columns)
assert len(cov) > 0, "empty coverage"
print("[smoke] PASS — parquets written; coverage rows:", len(cov))
PY
echo "[smoke_pixel_profile] PASS"
