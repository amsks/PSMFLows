#!/usr/bin/env bash
# BASELINE: PSM in RAW ACTION SPACE with NO BC ANCHORING.  cube-single-play, 3 seeds, 500k.
#
# WHAT THIS IS. The primary agent (`agent=psmflow`) is the paper-strict AFFINE
# LatentFlowPSM: the successor measure is affine in the task coordinate and every action it
# evaluates is a decode of the frozen behaviour flow. This baseline strips the flow out and
# keeps everything else: the same affine measure, the same proto/codebook TD objective, the
# same amortized w-conditioned actor -- but over RAW actions, with NO preimages and NO
# behaviour-cloning anchor on the actor. It is the control for "does the latent action
# interface buy anything, or would raw-action PSM do as well?".
#
#   arm A (primary)   agent=affine_psm   M(s,a,x) = Phi(s,a,x).w + b(s,a,x), factored
#                     Phi = A(s,a) phi_x(x)  -- the RAW-ACTION twin of the affine psmflow.
#   arm B (secondary) agent=psm          M(s,a,x) = psi(s,z,a)^T phi(x) -- the bilinear PSM
#                     of arXiv 2411.19418, the peer baseline, also raw-action.
#
# BC OFF. `agent.actor.bc_coeff=0.0` is the single switch in both agents. At 0 the actor
# loss is exactly `-Q.mean()` with Q = Phi(s,a,g).w_g + b (arm A) / psi(s,z,a)^T z (arm B):
# no MSE-to-data-action term, no flow distillation target, and the |Q| normalisation that
# only exists to scale the BC term against Q is skipped too. `actor.type=ddpgbc` is used so
# that the objective is literally `-Q` -- the `flow` actor would keep training a
# flow-matching velocity field on dataset actions (dead compute once nothing distils from
# it, and a BC loss still sitting in the printed objective).
#
# PESSIMISM KEPT AS-IS. Arm A has no pessimism penalty at all; its only guards are the
# sqrt(d)-normalised Phi/w, the tanh-bounded offset (b_scale=10), ortho_coef=1000 and the
# tau=0.01 target nets. Arm B keeps the agent's own defaults: pessimism_penalty=0.0 on the
# TD target and actor_pessimism_penalty=0.5 (ensemble-disagreement penalty, num_parallel=2)
# on the actor's Q. Nothing was added or removed.
#
# EXPECTED FAILURE MODE, stated up front: with no anchor an offline actor is free to walk
# off the data manifold and exploit the critic, so low success with a large actor_q is the
# predicted outcome, not a bug.
#
# Usage:
#   PSM_REPO=<checkout> PSM_DATA=<scratch> bash scripts/baselines/psm_raw_nobc.sh train
#   PSM_REPO=... PSM_DATA=... RUN_ROOT=<group dir> bash scripts/baselines/psm_raw_nobc.sh eval
set -euo pipefail

: "${PSM_REPO:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${PSM_DATA:=/mnt/home/amohan/psm-data}"
export PSM_REPO PSM_DATA
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}"

AGENT="${AGENT:-affine_psm}"
GROUP="${GROUP:-psm_raw_nobc_cube}"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-500000}"
SAVE_INT="${SAVE_INT:-50000}"
PART="${PART:-kisski-inference}"
ACCT="${ACCT:-general}"
# The BC-off arm, in full. `actor.type=ddpgbc` + `bc_coeff=0` == loss is -Q.mean().
NOBC_EXTRA="${NOBC_EXTRA:-agent.actor.type=ddpgbc agent.actor.bc_coeff=0.0}"

case "${1:-train}" in
  train)
    for s in $SEEDS; do
      sbatch --partition="$PART" --account="$ACCT" --time=08:00:00 \
        --job-name="tr_${GROUP}_sd${s}" --output="$PSM_DATA/logs/baselines/%x-%j.out" \
        --export=ALL,AGENT="$AGENT",ENVKEY=cube,SEED="$s",GROUP="$GROUP",STEPS="$STEPS",\
EVAL_INT=50000,EVAL_EPS=50,LOG_INT=5000,SAVE_INT="$SAVE_INT",EXTRA="$NOBC_EXTRA" \
        "$PSM_REPO/scripts/baselines/train_archived.sbatch"
    done
    ;;
  eval)
    # RUN_ROOT = $PSM_DATA/exp/PSMFLows/<GROUP>; one 500-episode job per (seed, epoch).
    : "${RUN_ROOT:?RUN_ROOT=<group dir under exp/PSMFLows> required}"
    i=0
    for d in "$RUN_ROOT"/sd*; do
      s=$(basename "$d" | sed 's/^sd0*\([0-9]\)_.*/\1/')
      for e in 50000 100000 150000 200000 250000 300000 350000 400000 450000 500000; do
        [ -f "$d/params_$e.pkl" ] || { echo "skip: no $d/params_$e.pkl"; continue; }
        k=$((e / 1000))
        sbatch --partition="$PART" --account="$ACCT" --time=03:00:00 \
          --job-name="e500_${GROUP}_${k}k_sd${s}" --output="$PSM_DATA/logs/baselines/%x-%j.out" \
          --export=ALL,AGENT="$AGENT",ENVKEY=cube,RUN_DIR="$d",EPOCH="$e",\
OUT="eval500_${GROUP}_${k}k_sd${s}" \
          "$PSM_REPO/scripts/baselines/eval_archived.sbatch"
        i=$((i + 1))
      done
    done
    echo "submitted $i eval jobs"
    ;;
  *) echo "usage: $0 train|eval" >&2; exit 1 ;;
esac
