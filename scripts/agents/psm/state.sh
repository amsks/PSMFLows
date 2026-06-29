#!/usr/bin/env bash
# scripts/agents/psm/state.sh — PSM+FlowBC on cube_single (state).
# Universal knobs: SEEDS, GPUS, TRAIN_STEPS, RUN_GROUP, DATA_PATH, DRY_RUN.
# PSM knobs (appended to the hydra extra args):
#   Z_DIM         continuous z / phi-basis dimension (default = agent config)
#   MAX_LOG_SEED  proto (binary) z dimension
#   NUM_PARALLEL  number of parallel SF/PSM psi heads (sf.num_parallel)
USAGE="usage: bash scripts/agents/psm/state.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'   space-separated seeds
    GPUS='0 1'          space-separated GPU ids (seed i -> GPUS[i % nGPU])
    TRAIN_STEPS=1000000
    RUN_GROUP=psm_state wandb group + log dir name
    DATA_PATH=/dev/shm/factored-fb/datasets
    Z_DIM=             continuous z dim (default from agent config)
    MAX_LOG_SEED=      proto binary z dim
    NUM_PARALLEL=      sf.num_parallel
    SAVE_EVAL_VIDEOS=false  forward to train.py as save_eval_videos=<val>
    DRY_RUN=1           print commands instead of launching"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

RUN_GROUP="${RUN_GROUP:-psm_state}"

extra=("agent=psm_flowbc")
[[ -n "${Z_DIM:-}" ]]        && extra+=("z_dim=$Z_DIM")
[[ -n "${MAX_LOG_SEED:-}" ]] && extra+=("max_log_seed=$MAX_LOG_SEED")
[[ -n "${NUM_PARALLEL:-}" ]] && extra+=("sf.num_parallel=$NUM_PARALLEL")

_run_fb_sweep cube_single "$RUN_GROUP" "${extra[@]}"
