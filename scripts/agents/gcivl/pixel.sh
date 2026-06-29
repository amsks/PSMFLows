#!/usr/bin/env bash
# scripts/agents/gcivl/pixel.sh — GCIVL on visual_cube_single (pixel only — no state config).
# ENCODER=impala (default) | drq.
USAGE="usage: bash scripts/agents/gcivl/pixel.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'
    GPUS='0 1'
    TRAIN_STEPS=500000
    RUN_GROUP=gcivl_pixel
    ENCODER=impala               impala | drq
    WANDB_MODE=online
    STORAGE=shm
    DRY_RUN=1"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

ENCODER="${ENCODER:-impala}"
case "$ENCODER" in
    impala) CONFIG=cube_single_visual_gcivl ;;
    drq)    CONFIG=cube_single_visual_gcivl_drq ;;
    *) echo "ENCODER must be impala|drq (got: $ENCODER)" >&2; exit 1 ;;
esac

RUN_GROUP="${RUN_GROUP:-gcivl_pixel_${ENCODER}}"
TRAIN_STEPS="${TRAIN_STEPS:-500000}"
export TRAIN_STEPS
_run_jax_sweep "$CONFIG" "$RUN_GROUP"
