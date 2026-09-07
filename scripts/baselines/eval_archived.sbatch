#!/usr/bin/env bash
# One SLURM job = one 500-episode eval of one ARCHIVED-agent checkpoint.
#
# Submit:
#   sbatch --export=ALL,AGENT=affine_psm,ENVKEY=cube,RUN_DIR=<dir>,EPOCH=500000,OUT=<basename> \
#     scripts/baselines/eval_archived.sbatch
#SBATCH --job-name=arch-e500
# SLURM does NOT expand environment variables in #SBATCH directives, so this is the
# literal $PSM_DATA/logs/slurm on KISSKI rather than the variable. A relative path here
# lands the log in the submit CWD -- i.e. the repo root. Launchers override both flags.
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

: "${PSM_REPO:?PSM_REPO required}"
: "${PSM_DATA:?PSM_DATA required}"
: "${AGENT:?AGENT=psm|affine_psm required}"
: "${ENVKEY:?ENVKEY=cube|antmaze|pointmaze required}"
: "${RUN_DIR:?RUN_DIR=<training run dir> required}"
: "${OUT:?OUT=<report json basename> required}"
EPOCH="${EPOCH:-500000}"
EPISODES="${EPISODES:-500}"
EXTRA="${EXTRA:-}"

case "$ENVKEY" in
  cube)      ENV_NAME=cube-single-play-singletask-v0 ;;
  antmaze)   ENV_NAME=antmaze-medium-navigate-singletask-v0 ;;
  pointmaze) ENV_NAME=pointmaze-medium-navigate-singletask-task1-v0 ;;
  *) echo "unknown ENVKEY: $ENVKEY" >&2; exit 1 ;;
esac

cd "$PSM_REPO"
PY="${PY:-$PSM_REPO/.venv/bin/python}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRAC:-0.90}"
export MUJOCO_GL=egl
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}"
mkdir -p "$PSM_DATA/logs"

echo "=== eval500 (archived): agent=$AGENT env=$ENV_NAME epoch=$EPOCH out=$OUT"
echo "    run_dir=$RUN_DIR"

# shellcheck disable=SC2086
srun "$PY" scripts/baselines/eval_archived.py \
    agent="$AGENT" env_name="$ENV_NAME" \
    restore_path="$RUN_DIR" restore_epoch="$EPOCH" \
    eval_episodes="$EPISODES" report_out="$PSM_DATA/logs/$OUT.json" $EXTRA
