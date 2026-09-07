#!/usr/bin/env bash
# Critic-ensemble disagreement vs distance-to-data of the policy index u', one checkpoint
# per SLURM job (tools/diag_ensemble_disagreement.py). Read-only.
#
# Submit:
#   sbatch --partition=kisski-inference --account=general --time=01:00:00 \
#     --job-name=disag_<tag> --output=$PSM_DATA/logs/%x-%j.out \
#     --export=ALL,PSM_REPO=...,PSM_DATA=...,ENVKEY=antmaze,RUN_DIR=<run dir>,EPOCH=100000,TAG=<name> \
#     scripts/slurm/diag_ensemble_disagreement.sbatch
#
# Writes $PSM_DATA/logs/diag_disagreement/<TAG>_<EPOCH>.json.
#SBATCH --job-name=disag
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

: "${PSM_REPO:?PSM_REPO=<path to the PSMFLows checkout> required}"
: "${PSM_DATA:?PSM_DATA=<scratch dir with flow/ and preimages/> required}"
: "${ENVKEY:?ENVKEY=cube|antmaze|pointmaze|scene required}"
: "${RUN_DIR:?RUN_DIR=<stage-C run dir with flags.json and params_*.pkl> required}"
: "${EPOCH:?EPOCH=<checkpoint epoch> required}"
TAG="${TAG:-$ENVKEY}"
N_BATCHES="${N_BATCHES:-8}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
KNN_SUBSET="${KNN_SUBSET:-20000}"
EXTRA="${EXTRA:-}"

case "$ENVKEY" in
  cube)      ENV_NAME=cube-single-play-singletask-v0;                FLOW=cube-single-play ;;
  antmaze)   ENV_NAME=antmaze-medium-navigate-singletask-v0;         FLOW=antmaze-medium-navigate ;;
  pointmaze) ENV_NAME=pointmaze-medium-navigate-singletask-task1-v0; FLOW=pointmaze-medium-navigate ;;
  scene)     ENV_NAME=scene-play-singletask-v0;                     FLOW=scene-play ;;
  *) echo "unknown ENVKEY: $ENVKEY" >&2; exit 1 ;;
esac

FLOW_DIR="$PSM_DATA/flow/$FLOW"
PRE_NPZ="$PSM_DATA/preimages/$FLOW.npz"
LOGS="$PSM_DATA/logs/diag_disagreement"
OUT="$LOGS/${TAG}_${EPOCH}.json"
[ -d "$FLOW_DIR" ] || { echo "no such flow dir: $FLOW_DIR" >&2; exit 1; }
[ -f "$PRE_NPZ" ]  || { echo "no such preimage npz: $PRE_NPZ" >&2; exit 1; }
[ -f "$RUN_DIR/params_${EPOCH}.pkl" ] || { echo "no params_${EPOCH}.pkl in $RUN_DIR" >&2; exit 1; }
mkdir -p "$LOGS"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRAC:-0.90}"
export MUJOCO_GL=egl
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}"

echo "=== diag_ensemble_disagreement: env=$ENVKEY epoch=$EPOCH tag=$TAG"
echo "    run_dir=$RUN_DIR  n_batches=$N_BATCHES batch_size=$BATCH_SIZE knn_subset=$KNN_SUBSET"
echo "    out=$OUT"

cd "$PSM_REPO"
# shellcheck disable=SC2086  -- EXTRA is a deliberate word-split list of hydra overrides
srun .venv/bin/python tools/diag_ensemble_disagreement.py agent=psmflow \
  env_name="$ENV_NAME" \
  agent.flow_ckpt_path="$FLOW_DIR" agent.flow_ckpt_epoch=500000 \
  agent.preimage_path="$PRE_NPZ" \
  restore_path="$RUN_DIR" restore_epoch="$EPOCH" \
  +n_batches="$N_BATCHES" +batch_size="$BATCH_SIZE" +knn_subset="$KNN_SUBSET" \
  report_out="$OUT" $EXTRA
