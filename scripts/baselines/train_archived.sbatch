#!/usr/bin/env bash
# One SLURM job = one seed of one ARCHIVED agent (scripts/baselines/run_archived.py).
# The archive-aware twin of scripts/slurm/train_psmflow.sbatch: no flow checkpoint, no
# preimage npz -- these baselines act in RAW action space.
#
# Submit:
#   sbatch --partition=kisski-inference --account=general \
#     --export=ALL,AGENT=affine_psm,ENVKEY=cube,SEED=0,GROUP=psm_raw_nobc_cube,EXTRA="agent.actor.bc_coeff=0.0" \
#     scripts/baselines/train_archived.sbatch
#SBATCH --job-name=arch-train
# SLURM does NOT expand environment variables in #SBATCH directives, so this is the
# literal $PSM_DATA/logs/slurm on KISSKI rather than the variable. A relative path here
# lands the log in the submit CWD -- i.e. the repo root. Launchers override both flags.
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

: "${PSM_REPO:?PSM_REPO=<path to the PSMFLows checkout> required}"
: "${PSM_DATA:?PSM_DATA=<scratch dir> required}"
: "${AGENT:?AGENT=psm|affine_psm required}"
: "${ENVKEY:?ENVKEY=cube|antmaze|pointmaze required}"
SEED="${SEED:-0}"
STEPS="${STEPS:-500000}"
EVAL_INT="${EVAL_INT:-50000}"
EVAL_EPS="${EVAL_EPS:-50}"
LOG_INT="${LOG_INT:-5000}"
SAVE_INT="${SAVE_INT:-50000}"
EXTRA="${EXTRA:-}"

case "$ENVKEY" in
  cube)      ENV_NAME=cube-single-play-singletask-v0 ;;
  antmaze)   ENV_NAME=antmaze-medium-navigate-singletask-v0 ;;
  pointmaze) ENV_NAME=pointmaze-medium-navigate-singletask-task1-v0 ;;
  *) echo "unknown ENVKEY: $ENVKEY" >&2; exit 1 ;;
esac
GROUP="${GROUP:-${AGENT}_${ENVKEY}_$(date +%Y%m%d_%H%M%S)}"

cd "$PSM_REPO"
PY="${PY:-$PSM_REPO/.venv/bin/python}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRAC:-0.90}"
export MUJOCO_GL=egl
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$PSM_DATA/wandb"
mkdir -p "$PSM_DATA/exp" "$PSM_DATA/wandb"

echo "=== archived baseline: agent=$AGENT $ENV_NAME seed=$SEED group=$GROUP ==="
echo "    steps=$STEPS save_interval=$SAVE_INT extra='$EXTRA'"

# shellcheck disable=SC2086  -- EXTRA is a deliberate word-split list of hydra overrides
srun "$PY" scripts/baselines/run_archived.py \
    agent="$AGENT" $EXTRA \
    env_name="$ENV_NAME" \
    offline_steps="$STEPS" online_steps=0 \
    log_interval="$LOG_INT" eval_interval="$EVAL_INT" eval_episodes="$EVAL_EPS" \
    save_interval="$SAVE_INT" save_dir="$PSM_DATA/exp" \
    run_group="$GROUP" seed="$SEED"
