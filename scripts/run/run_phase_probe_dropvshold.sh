#!/usr/bin/env bash
# Re-run state phase_probe on S0 with extended signals (final_cube_lift, final_grip)
# to disambiguate dropped-cube vs held-but-mispositioned transport failures.
# Outputs go to analysis/probes/phase_probe_v2/sN_final/ (old runs preserved).
set -euo pipefail

cd "$(dirname "$0")/.."

SEEDS=(3 4 5 6 7 8 10)
RUN_ROOT="results/Factored-FB-cube-run"
RUN_PREFIX="2026-05-14_21-44-19__cube-single-play-v0__fb_flowbc__s"
OUT_ROOT="analysis/probes/phase_probe_v2"
DATA_PATH="datasets"

mkdir -p "$OUT_ROOT"
for s in "${SEEDS[@]}"; do
  run_dir="$RUN_ROOT/${RUN_PREFIX}${s}"
  cfg="$run_dir/.hydra/config.yaml"
  ckpt="$run_dir/checkpoints/final.pt"
  out="$OUT_ROOT/s${s}_final"
  if [[ ! -f "$cfg" || ! -f "$ckpt" ]]; then
    echo "[skip] s$s: missing cfg or ckpt"
    continue
  fi
  echo "[probe] s$s -> $out"
  MUJOCO_GL=glfw .venv/bin/python -m scripts.probes.phase_probe \
    --config "$cfg" --checkpoint "$ckpt" \
    --out "$out" --device cpu \
    --n-episodes 10 --scenarios S0 \
    --data-path "$DATA_PATH" \
    > "$out.log" 2>&1
done
echo "[done] all seeds processed -> $OUT_ROOT"
