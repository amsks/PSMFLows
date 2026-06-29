#!/usr/bin/env bash
# scripts/agents/rldp/pixel.sh — RLDP+FlowBC on visual_cube_single (pixel obs, frame-stack 3).
# Selects the visual domain + DrQ encoder (via the domain config) and the rldp_flowbc
# agent config group. OGBench visual default is 500k steps.
USAGE="usage: bash scripts/agents/rldp/pixel.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'
    GPUS='0 1'
    TRAIN_STEPS=500000           (OGBench visual default)
    RUN_GROUP=rldp_pixel
    DATA_PATH=/dev/shm/factored-fb/datasets
    Z_DIM=             continuous z dim (default from agent config)
    HORIZON=           SP-loss window length (default 3 from domain config)
    SAVE_EVAL_VIDEOS=false       forward to train.py as save_eval_videos=<val>
    DRY_RUN=1"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

RUN_GROUP="${RUN_GROUP:-rldp_pixel}"
TRAIN_STEPS="${TRAIN_STEPS:-500000}"

# visual_cube_single_rldp selects domain=visual + drq encoder + agent=rldp_flowbc.
extra=()
[[ -n "${Z_DIM:-}" ]]   && extra+=("z_dim=$Z_DIM")
[[ -n "${HORIZON:-}" ]] && extra+=("horizon=$HORIZON")

export TRAIN_STEPS
_run_fb_sweep visual_cube_single_rldp "$RUN_GROUP" "${extra[@]}"
