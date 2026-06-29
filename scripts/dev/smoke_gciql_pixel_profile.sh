#!/usr/bin/env bash
# Local smoke for the GCIQL pixel profiler. Rolls out the real sd000 visual
# checkpoint (jax venv) for a couple of short episodes and verifies the four
# parquets + cube-space coverage. The coverage reference uses cube xyz from the
# local STATE dataset (physics is modality-independent), so no pixel dataset is
# needed. Verifies the obs_type-aware + impala-encoder GCIQL pixel path.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

JAX_PY="${JAX_PY:-.venv-jax-cpu/bin/python}"
PY="${PY:-.venv/bin/python}"
GCIQL_ROOT="${GCIQL_ROOT:-RESULTS/gciql-pixel-results}"
STATE_DS="${STATE_DS:-datasets/cube-single-play-v0}"
MUJOCO_GL="${MUJOCO_GL:-glfw}"
STEP="${STEP:-500000}"

D=$(ls -d "$GCIQL_ROOT"/sd000_* 2>/dev/null | head -1)
[ -n "$D" ] || { echo "FAIL: no sd000 GCIQL pixel run under $GCIQL_ROOT"; exit 1; }
[ -d "$STATE_DS/buffer" ] || { echo "FAIL: no local state dataset at $STATE_DS"; exit 1; }
[ -x "$JAX_PY" ] || { echo "FAIL: jax venv python not found at $JAX_PY"; exit 1; }

OUT=$(mktemp -d)/sd000_final
MUJOCO_GL="$MUJOCO_GL" "$JAX_PY" scripts/profiles/gciql_profile.py \
    --run-dir "$D" --step "$STEP" --out "$OUT" \
    --obs-type pixels --dataset-path "$STATE_DS" \
    --tasks 1 --n-episodes 1 --max-steps 50

"$PY" - "$OUT" <<'PY'
import sys
import pandas as pd
from pathlib import Path
out = Path(sys.argv[1])
for name in ("value_landscape", "value_steps", "coverage", "phase_funnel"):
    p = out / f"{name}.parquet"
    assert p.exists(), f"missing {p}"
cov = pd.read_parquet(out / "coverage.parquet")
assert {"task", "outcome", "region", "nn_dist"} <= set(cov.columns), list(cov.columns)
assert len(cov) > 0, "empty coverage"
print("[smoke] PASS — GCIQL pixel parquets written; coverage rows:", len(cov))
PY
echo "[smoke_gciql_pixel_profile] PASS"
