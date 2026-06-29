#!/usr/bin/env bash
# scripts/agents/psm/pixel.sh — PSM+FlowBC on visual_cube_single (pixel obs, frame-stack 3).
# Selects the visual domain + DrQ encoder (via the domain config) and swaps the agent
# config group to psm_flowbc. OGBench visual default is 500k steps.
# PSM knobs (appended to the hydra extra args):
#   Z_DIM / MAX_LOG_SEED / NUM_PARALLEL  (see state.sh)
USAGE="usage: bash scripts/agents/psm/pixel.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'
    GPUS='0 1'
    TRAIN_STEPS=500000           (OGBench visual default)
    RUN_GROUP=psm_pixel
    DATA_PATH=/dev/shm/factored-fb/datasets
    Z_DIM=             continuous z dim (default from agent config)
    MAX_LOG_SEED=      proto binary z dim
    NUM_PARALLEL=      sf.num_parallel
    SAVE_EVAL_VIDEOS=false       forward to train.py as save_eval_videos=<val>
    DRY_RUN=1"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

RUN_GROUP="${RUN_GROUP:-psm_pixel}"
TRAIN_STEPS="${TRAIN_STEPS:-500000}"

# visual_cube_single_psm selects domain=visual + drq encoder + agent=psm_flowbc.
extra=()
[[ -n "${Z_DIM:-}" ]]        && extra+=("z_dim=$Z_DIM")
[[ -n "${MAX_LOG_SEED:-}" ]] && extra+=("max_log_seed=$MAX_LOG_SEED")
[[ -n "${NUM_PARALLEL:-}" ]] && extra+=("sf.num_parallel=$NUM_PARALLEL")

export TRAIN_STEPS
_run_fb_sweep visual_cube_single_psm "$RUN_GROUP" "${extra[@]}"
