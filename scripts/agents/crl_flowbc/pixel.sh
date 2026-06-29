#!/usr/bin/env bash
# scripts/agents/crl_flowbc/pixel.sh — CRL + FlowBC actor on visual_cube_single (DrQ pixel front-end).
USAGE="usage: bash scripts/agents/crl_flowbc/pixel.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'
    GPUS='0 1'
    TRAIN_STEPS=500000
    RUN_GROUP=crl_flowbc_pixel
    WANDB_MODE=online
    STORAGE=shm
    DRY_RUN=1"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

RUN_GROUP="${RUN_GROUP:-crl_flowbc_pixel}"
TRAIN_STEPS="${TRAIN_STEPS:-500000}"
export TRAIN_STEPS
_run_jax_sweep cube_single_visual_crl_flowbc_drq "$RUN_GROUP"
