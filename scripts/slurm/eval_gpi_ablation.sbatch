#!/usr/bin/env bash
# One `agent.gpi_select` ablation eval per SLURM job -- the arms of
# docs/design/2026-09-05-gpi-ablations.md, which ask whether the affine cube run's success
# comes from the critic or from the large-norm bias any max over K prior draws produces.
#
# It calls tools/eval_checkpoint.py directly rather than going through scripts/eval500.sh
# because the smoke pass needs EPISODES != 500 and eval500.sh hardcodes that. Everything
# else is the same protocol: the agent config comes from the run's own flags.json, with
# the typed `agent.gpi_select=...` (and friends) layered on top. `gpi_select` did not
# exist when these runs were trained, so it is absent from their flags.json and falls back
# to this checkout's default `argmax` = the shipped rule -- which is why MODE=argmax is a
# reproduction control, not a no-op.
#
# Submit (all arms concurrently):
#   sbatch --partition=kisski-inference --account=general --time=01:00:00 \
#     --job-name=gpiabl_small_ball_350000 --output=$PSM_DATA/logs/%x-%j.out \
#     --export=ALL,RUN_DIR=<run dir>,EPOCH=350000,MODE=small_ball,TAG=gpiabl \
#     scripts/slurm/eval_gpi_ablation.sbatch
#
# Writes $PSM_DATA/logs/<TAG>_<NAME>_<EPOCH>.json (NAME defaults to MODE).
#SBATCH --job-name=gpiabl
# SLURM does NOT expand environment variables in #SBATCH directives, so this is the
# literal $PSM_DATA/logs/slurm on KISSKI rather than the variable. A relative path here
# lands the log in the submit CWD -- i.e. the repo root. Launchers override both flags.
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

: "${PSM_REPO:?PSM_REPO=<path to the PSMFLows checkout> required}"
: "${PSM_DATA:?PSM_DATA=<scratch dir with flow/ and preimages/> required}"
: "${RUN_DIR:?RUN_DIR=<stage-C run dir with flags.json and params_*.pkl> required}"
: "${EPOCH:?EPOCH=<checkpoint epoch> required}"
MODE="${MODE:-argmax}"
NAME="${NAME:-$MODE}"
TAG="${TAG:-gpiabl}"
ENVKEY="${ENVKEY:-cube}"
EPISODES="${EPISODES:-500}"
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
LOGS="$PSM_DATA/logs"
OUT="$LOGS/${TAG}_${NAME}_${EPOCH}.json"
[ -d "$FLOW_DIR" ] || { echo "no such flow dir: $FLOW_DIR" >&2; exit 1; }
[ -f "$PRE_NPZ" ]  || { echo "no such preimage npz: $PRE_NPZ" >&2; exit 1; }
[ -f "$RUN_DIR/params_${EPOCH}.pkl" ] || { echo "no params_${EPOCH}.pkl in $RUN_DIR" >&2; exit 1; }
mkdir -p "$LOGS"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRAC:-0.90}"
export MUJOCO_GL=egl
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}"

echo "=== eval_gpi_ablation: env=$ENVKEY epoch=$EPOCH gpi_select=$MODE episodes=$EPISODES"
echo "    run_dir=$RUN_DIR"
echo "    extra='$EXTRA'  out=$OUT"

cd "$PSM_REPO"
# shellcheck disable=SC2086  -- EXTRA is a deliberate word-split list of hydra overrides
srun .venv/bin/python tools/eval_checkpoint.py agent=psmflow \
  env_name="$ENV_NAME" \
  agent.flow_ckpt_path="$FLOW_DIR" agent.flow_ckpt_epoch=500000 \
  agent.preimage_path="$PRE_NPZ" agent.use_point_preimage=true \
  agent.gpi_select="$MODE" \
  restore_path="$RUN_DIR" restore_epoch="$EPOCH" \
  eval_episodes="$EPISODES" report_out="$OUT" $EXTRA
