#!/usr/bin/env bash
# scripts/agents/rldp_flowbc/state.sh — RLDP + FlowBC on cube_single (state).
USAGE="usage: bash scripts/agents/rldp_flowbc/state.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'
    GPUS='0 1'
    TRAIN_STEPS=1000000
    RUN_GROUP=rldp_flowbc_state
    DATA_PATH=/dev/shm/factored-fb/datasets
    DRY_RUN=1"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

RUN_GROUP="${RUN_GROUP:-rldp_flowbc_state}"
_run_fb_sweep cube_single "$RUN_GROUP" agent=rldp_flowbc
