#!/usr/bin/env bash
# scripts/agents/fb/state.sh — FB+FlowBC on cube_single (state).
# Variants via env vars:
#   MODE=vanilla|onestep             (default vanilla)
#   REWEIGHT_ALPHA=<float>           (default 0; >0 enables coverage-balanced reweight)
# Universal knobs: SEEDS, GPUS, TRAIN_STEPS, RUN_GROUP, DATA_PATH, DRY_RUN.
USAGE="usage: bash scripts/agents/fb/state.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4'   space-separated seeds
    GPUS='0 1'          space-separated GPU ids (seed i -> GPUS[i % nGPU])
    TRAIN_STEPS=1000000
    RUN_GROUP=fb_state  wandb group + log dir name
    DATA_PATH=/dev/shm/factored-fb/datasets
    MODE=vanilla        vanilla | onestep
    REWEIGHT_ALPHA=     >0 enables coverage-balanced reweight (needs analysis/cube_density.npz)
    SAVE_EVAL_VIDEOS=false  forward to train.py as save_eval_videos=<val>
    DRY_RUN=1           print commands instead of launching"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

RUN_GROUP="${RUN_GROUP:-fb_state}"
MODE="${MODE:-vanilla}"
REWEIGHT_ALPHA="${REWEIGHT_ALPHA:-0}"
ORTHO_COEF="${ORTHO_COEF:-1000}"
LR_B="${LR_B:-1e-4}"
L_DIM="${L_DIM:-}"            # empty = config default (50); set 256 for the capacity ablation
GOAL_COND="${GOAL_COND:-false}"
FIXED_B="${FIXED_B:-none}"    # none | cube_xyz

extra=("ortho_coef=$ORTHO_COEF" "lr_b=$LR_B")
[[ "$MODE" == "onestep" ]] && extra+=("onestep=true")
[[ -n "$L_DIM" ]] && extra+=("L_dim=$L_DIM")
[[ "$GOAL_COND" == "true" ]] && extra+=("goal_cond=true")
[[ "$FIXED_B" != "none" ]] && extra+=("fixed_b=$FIXED_B")
if [[ "$REWEIGHT_ALPHA" != "0" ]]; then
    DENSITY="${DENSITY:-$REPO/analysis/cube_density.npz}"
    extra+=("reweight_alpha=$REWEIGHT_ALPHA"
            "reweight_clip=${REWEIGHT_CLIP:-10.0}"
            "reweight_density_path=$DENSITY"
            "weight_diag=true" "weight_z=true")
fi

_run_fb_sweep cube_single "$RUN_GROUP" "${extra[@]}"
