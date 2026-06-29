#!/usr/bin/env bash
# Re-run state phase_probe on FB-pixel and GCIQL-pixel-DrQ checkpoints with
# the extended classify_phases output (final_cube_lift, final_grip) so we can
# split transport-fails into grasp-lost vs misplaced for both methods on pixels.
# Outputs:
#   FB pixel:   analysis/probes/phase_probe_pixel_v2/sN_final/
#   GCIQL DrQ:  analysis/profiles/gciql_profile_pixel_drq_v2/sN_final/
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
JAX_PY="${JAX_PY:-.venv-jax-cpu/bin/python}"

FB_ROOT="results/fb-pixel-results"
FB_OUT="analysis/probes/phase_probe_pixel_v2"
FB_CKPT_STEP="600000"
FB_DATA="datasets_pixel"

GCIQL_ROOT="results/factored-fb-gciql-pixel-drq"
GCIQL_OUT="analysis/profiles/gciql_profile_pixel_drq_v2"
GCIQL_STEP="500000"
GCIQL_DATA="datasets_pixel/cube-single-play-v0"

MUJOCO_GL="glfw"
N_EPISODES=10

mkdir -p "$FB_OUT" "$GCIQL_OUT"

# --- FB pixel phase probe (S0 only; pixel S1/S2 not supported) ---
shopt -s nullglob
for d in "$FB_ROOT"/*__s*; do
  [ -d "$d" ] || continue
  s=$(basename "$d" | sed -E 's/.*__s([0-9]+)$/\1/')
  cfg="$d/.hydra/config.yaml"
  ckpt="$d/checkpoints/step_${FB_CKPT_STEP}.pt"
  out="$FB_OUT/s${s}_final"
  if [[ ! -f "$cfg" || ! -f "$ckpt" ]]; then
    echo "[skip] FB s$s: missing cfg or step_${FB_CKPT_STEP} ckpt"
    continue
  fi
  if [[ -f "$out/per_episode.parquet" ]]; then
    echo "[skip] FB s$s already done"
    continue
  fi
  echo "[FB pixel] s$s -> $out"
  MUJOCO_GL=$MUJOCO_GL "$PY" -m scripts.probes.phase_probe \
    --config "$cfg" --checkpoint "$ckpt" \
    --out "$out" --device cpu \
    --n-episodes "$N_EPISODES" --scenarios S0 \
    --data-path "$FB_DATA" \
    > "$out.log" 2>&1
done

# --- GCIQL pixel DrQ profile (jax venv) ---
# Need tools/wandb_mode_shim on PYTHONPATH so sitecustomize registers the
# 'drq' encoder before AgentCls.create reads config['encoder'] (otherwise
# encoder_modules KeyError 'drq'). Also need third_party/ogbench/impls so the
# shim's `import utils.encoders` succeeds at interpreter startup.
GCIQL_PY_PATH="tools/wandb_mode_shim:third_party/ogbench/impls"
for d in "$GCIQL_ROOT"/sd*; do
  [ -d "$d" ] || continue
  s=$(basename "$d" | sed -E 's/^sd0*([0-9]+)_.*/\1/')
  out="$GCIQL_OUT/s${s}_final"
  if [[ -f "$out/phase_funnel.parquet" ]]; then
    echo "[skip] GCIQL s$s already done"
    continue
  fi
  echo "[GCIQL pixel DrQ] s$s <- $(basename "$d") -> $out"
  PYTHONPATH="$GCIQL_PY_PATH" MUJOCO_GL=$MUJOCO_GL "$JAX_PY" -m scripts.profiles.gciql_profile \
    --run-dir "$d" --step "$GCIQL_STEP" \
    --out "$out" --tasks 1,2,3,4,5 --n-episodes "$N_EPISODES" \
    --obs-type pixels --dataset-path "$GCIQL_DATA" \
    > "$out.log" 2>&1
done

echo "[done] FB pixel -> $FB_OUT ; GCIQL pixel DrQ -> $GCIQL_OUT"
