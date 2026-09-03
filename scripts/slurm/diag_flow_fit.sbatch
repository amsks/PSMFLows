#!/usr/bin/env bash
# D-flow-fit: does the frozen Stage-A behaviour flow FIT the data or MEMORIZE it?
# One env per job (~7 min on an H100 at 2048 anchors x 512 samples).
#
# Submit:
#   sbatch --partition=kisski-inference --account=general \
#     --job-name=flowfit-antmaze --output=$PSM_DATA/logs/%x-%j.out \
#     --export=ALL,ENVKEY=antmaze \
#     scripts/slurm/diag_flow_fit.sbatch
#
# ANCHORS/SAMPLES/KMAX default to the cube run's settings so envs stay comparable.
# The preimage npz is OPTIONAL -- it is read only for the `point_preimage_ceiling`
# line (the decode error of the exact inverse, i.e. the best any latent could do).
# Everything else samples u ~ N(0, I) and needs the decoder alone.
#SBATCH --job-name=flowfit
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
set -euo pipefail

PSM_REPO="${PSM_REPO:-/mnt/home/amohan/git/Austin/PSMFLows}"
PSM_DATA="${PSM_DATA:-/mnt/home/amohan/psm-data}"
: "${ENVKEY:?ENVKEY=cube|antmaze|pointmaze required}"
ANCHORS="${ANCHORS:-2048}"
SAMPLES="${SAMPLES:-512}"
KMAX="${KMAX:-512}"
EPOCH="${EPOCH:-500000}"
SEED="${SEED:-0}"

case "$ENVKEY" in
  cube)      ENV_NAME=cube-single-play-singletask-v0;                FLOW=cube-single-play ;;
  antmaze)   ENV_NAME=antmaze-medium-navigate-singletask-v0;         FLOW=antmaze-medium-navigate ;;
  pointmaze) ENV_NAME=pointmaze-medium-navigate-singletask-task1-v0; FLOW=pointmaze-medium-navigate ;;
  *) echo "unknown ENVKEY: $ENVKEY" >&2; exit 1 ;;
esac

FLOW_DIR="$PSM_DATA/flow/$FLOW"
[ -d "$FLOW_DIR" ] || { echo "no such flow dir: $FLOW_DIR" >&2; exit 1; }
OUT="${OUT:-$PSM_DATA/logs/diag_flow_fit_${ENVKEY}.json}"

# Ceiling line only; absent is fine.
PRE_NPZ="${PRE_NPZ:-$PSM_DATA/preimages/$FLOW.npz}"
PRE_ARG=()
if [ -f "$PRE_NPZ" ]; then PRE_ARG=("+fit_preimage=$PRE_NPZ"); else echo "no preimage npz at $PRE_NPZ -- ceiling line will be null"; fi

cd "$PSM_REPO"
PY="${PY:-$PSM_REPO/.venv/bin/python}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRAC:-0.90}"
export MUJOCO_GL=egl
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}"
export WANDB_MODE=offline
export WANDB_DIR="$PSM_DATA/wandb"
mkdir -p "$PSM_DATA/logs"

nvidia-smi -L || true
echo "=== flow fit: $ENV_NAME flow=$FLOW_DIR@$EPOCH anchors=$ANCHORS samples=$SAMPLES kmax=$KMAX"
echo "    out=$OUT  preimage=${PRE_ARG[*]:-<none>}"

srun "$PY" tools/diag_flow_fit.py \
    agent=fql \
    env_name="$ENV_NAME" \
    restore_path="$FLOW_DIR" restore_epoch="$EPOCH" \
    agent.flow_steps=100 \
    seed="$SEED" \
    +fit_anchors="$ANCHORS" +fit_samples="$SAMPLES" +fit_kmax="$KMAX" \
    ${PRE_ARG[@]+"${PRE_ARG[@]}"} \
    report_out="$OUT"
