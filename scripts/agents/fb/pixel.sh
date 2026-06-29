#!/usr/bin/env bash
# scripts/agents/fb/pixel.sh — FB+FlowBC on visual_cube_single (pixel obs, frame-stack 3).
# OGBench visual default is 500k steps. Variants: MODE=vanilla|onestep.
USAGE="usage: bash scripts/agents/fb/pixel.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'
    GPUS='0 1'
    TRAIN_STEPS=500000           (OGBench visual default)
    RUN_GROUP=fb_pixel
    DATA_PATH=/dev/shm/factored-fb/datasets
    MODE=vanilla                 vanilla | onestep
    SAVE_EVAL_VIDEOS=false       forward to train.py as save_eval_videos=<val>
    DRY_RUN=1"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

RUN_GROUP="${RUN_GROUP:-fb_pixel}"
TRAIN_STEPS="${TRAIN_STEPS:-500000}"
MODE="${MODE:-vanilla}"

extra=()
[[ "$MODE" == "onestep" ]] && extra+=("onestep=true")

export TRAIN_STEPS
_run_fb_sweep visual_cube_single "$RUN_GROUP" "${extra[@]}"
