#!/usr/bin/env bash
# One GPI-selection anatomy probe per SLURM job -- one job per checkpoint epoch, run
# concurrently, exactly as eval500.sbatch runs one eval per job.
#
# Submit (four epochs of the same run, concurrent):
#   for E in 250000 350000 400000 500000; do
#     sbatch --partition=kisski-inference --account=general --time=02:00:00 \
#       --job-name=gpisel_$E --output=$PSM_DATA/logs/%x-%j.out \
#       --export=ALL,ENVKEY=cube,RUN_DIR=<run dir>,EPOCH=$E,TAG=cube_sd0 \
#       scripts/slurm/diag_gpi_selection.sbatch
#   done
#
# Writes $PSM_DATA/logs/diag_gpi_selection_<TAG>_<EPOCH>.json plus a .npz sidecar holding
# the fixed-state Q tensor; `tools/diag_gpi_selection.py compare` turns the four npz files
# into the cross-checkpoint ranking-drift table (CPU, no GPU job needed).
#SBATCH --job-name=gpisel
# SLURM does NOT expand environment variables in #SBATCH directives, so this is the
# literal $PSM_DATA/logs/slurm on KISSKI rather than the variable. A relative path here
# lands the log in the submit CWD -- i.e. the repo root. Launchers override both flags.
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
#SBATCH --time=02:00:00
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
N_EP="${N_EP:-20}"
N_FIXED="${N_FIXED:-256}"
N_MC="${N_MC:-64}"
MC_H="${MC_H:-50}"
# The MC ground truth uses only the FROZEN flow, the dataset and the simulator, so every
# checkpoint's job would recompute the same numbers. ORACLE_DIR adds the third ranker (a
# frozen FQL expert's action critic read at G(s,u)); MC_CACHE makes the first job pay for
# the rollouts and the rest reuse them. Both optional.
ORACLE_DIR="${ORACLE_DIR:-}"
ORACLE_EPOCH="${ORACLE_EPOCH:-500000}"
MC_CACHE="${MC_CACHE:-}"
# NA_DIR: a trained DSRL-NA run whose qw/qa score the same roster as two more rankers.
NA_DIR="${NA_DIR:-}"
NA_EPOCH="${NA_EPOCH:-500000}"
EXTRA="${EXTRA:-}"
if [ -n "$ORACLE_DIR" ]; then
  EXTRA="$EXTRA +oracle_path=$ORACLE_DIR +oracle_epoch=$ORACLE_EPOCH"
fi
if [ -n "$MC_CACHE" ]; then
  EXTRA="$EXTRA +mc_cache=$MC_CACHE"
fi
if [ -n "$NA_DIR" ]; then
  EXTRA="$EXTRA +na_path=$NA_DIR +na_epoch=$NA_EPOCH"
fi
# The smaller box every ranker is ALSO scored inside (defaults to the NA run's own u_clip).
if [ -n "${INBOX_CLIP:-}" ]; then
  EXTRA="$EXTRA +inbox_clip=$INBOX_CLIP"
fi

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
OUT="$LOGS/diag_gpi_selection_${TAG}_${EPOCH}.json"
[ -d "$FLOW_DIR" ] || { echo "no such flow dir: $FLOW_DIR" >&2; exit 1; }
[ -f "$PRE_NPZ" ]  || { echo "no such preimage npz: $PRE_NPZ" >&2; exit 1; }
[ -f "$RUN_DIR/params_${EPOCH}.pkl" ] || { echo "no params_${EPOCH}.pkl in $RUN_DIR" >&2; exit 1; }
mkdir -p "$LOGS"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRAC:-0.90}"
export MUJOCO_GL=egl
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}"

echo "=== diag_gpi_selection: env=$ENVKEY epoch=$EPOCH tag=$TAG"
echo "    run_dir=$RUN_DIR"
echo "    n_episodes=$N_EP n_fixed_states=$N_FIXED n_mc_states=$N_MC mc_horizon=$MC_H"
echo "    out=$OUT"

cd "$PSM_REPO"
# shellcheck disable=SC2086  -- EXTRA is a deliberate word-split list of hydra overrides
srun .venv/bin/python tools/diag_gpi_selection.py agent=psmflow \
  env_name="$ENV_NAME" \
  agent.flow_ckpt_path="$FLOW_DIR" agent.flow_ckpt_epoch=500000 \
  agent.preimage_path="$PRE_NPZ" agent.use_point_preimage=true \
  restore_path="$RUN_DIR" restore_epoch="$EPOCH" \
  +n_episodes="$N_EP" +n_fixed_states="$N_FIXED" +n_mc_states="$N_MC" +mc_horizon="$MC_H" \
  report_out="$OUT" $EXTRA
