#!/usr/bin/env bash
# scripts/agents/gciql/state.sh — GCIQL on cube_single (state).
# OGBench's exact recipe: 1M steps, alpha=1.0 (config defaults).
USAGE="usage: bash scripts/agents/gciql/state.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'
    GPUS='0 1'
    TRAIN_STEPS=1000000
    RUN_GROUP=gciql_state
    WANDB_MODE=online            online | offline | disabled
    STORAGE=shm                  shm | nvme
    DRY_RUN=1"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

RUN_GROUP="${RUN_GROUP:-gciql_state}"
_run_jax_sweep cube_single_state "$RUN_GROUP"
