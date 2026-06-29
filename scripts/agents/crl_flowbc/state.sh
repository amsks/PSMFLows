#!/usr/bin/env bash
# scripts/agents/crl_flowbc/state.sh — CRL + FlowBC actor on cube_single (state).
USAGE="usage: bash scripts/agents/crl_flowbc/state.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'
    GPUS='0 1'
    TRAIN_STEPS=1000000
    RUN_GROUP=crl_flowbc_state
    WANDB_MODE=online
    STORAGE=shm
    DRY_RUN=1"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

RUN_GROUP="${RUN_GROUP:-crl_flowbc_state}"
_run_jax_sweep cube_single_state_crl_flowbc "$RUN_GROUP"
