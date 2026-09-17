#!/usr/bin/env bash
# The policy-side diagnostic (tools/diag_measure_vs_scalar_q.py, blocks G and H alongside
# A-F) on THREE cube checkpoints in ONE job, against one DSRL-NA scalar critic:
#   gpi       affine_strict_cube        (policy_index=latent, no actor; w from the reward file)
#   sec10     cube_sec10_affine_actor   (task_vector index, ddpg actor bc_coeff 1; own w)
#   sec10bc0  cube_sec10_affine_bc0     (task_vector index, ddpg actor bc_coeff 0; own w)
#
# Submit:
#   sbatch --partition=kisski-inference --account=general --time=00:45:00 \
#     --job-name=polside_cube_sd001 --output=$PSM_DATA/logs/%x-%j.out \
#     --export=ALL,PSM_REPO=<checkout>,GPI_DIR=...,SEC10_DIR=...,BC0_DIR=...,NA_DIR=...,\
#              REWARD_RAW=...,REWARD_SCALED=...,TAG=cube,SEED_TAG=sd001 \
#     scripts/slurm/diag_policy_side.sbatch
#
# Writes $PSM_DATA/logs/diag_policy_side_<TAG>_{gpi,sec10,sec10bc0}_<SEED_TAG>.json (+ .npz).
#SBATCH --job-name=polside
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

: "${PSM_REPO:?PSM_REPO=<path to the PSMFLows checkout> required}"
: "${PSM_DATA:?PSM_DATA=<scratch dir with flow/ and preimages/> required}"
: "${GPI_DIR:?GPI_DIR=<affine_strict_cube run dir> required}"
: "${SEC10_DIR:?SEC10_DIR=<cube_sec10_affine_actor run dir> required}"
: "${BC0_DIR:?BC0_DIR=<cube_sec10_affine_bc0 run dir> required}"
: "${NA_DIR:?NA_DIR=<DSRL-NA run dir> required}"
: "${REWARD_RAW:?REWARD_RAW=<raw r_hat npz> required}"
: "${REWARD_SCALED:?REWARD_SCALED=<scaled r_hat npz> required}"
EPOCH="${EPOCH:-500000}"
NA_EPOCH="${NA_EPOCH:-500000}"
TAG="${TAG:-cube}"
SEED_TAG="${SEED_TAG:-sd001}"
ENV_NAME=cube-single-play-singletask-v0
FLOW_DIR="$PSM_DATA/flow/cube-single-play"
PRE_NPZ="$PSM_DATA/preimages/cube-single-play.npz"
LOGS="$PSM_DATA/logs"
EXTRA="${EXTRA:-}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRAC:-0.90}"
export MUJOCO_GL=egl
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}"
cd "$PSM_REPO"

run_one () {  # name run_dir w_source
  local NAME="$1" RUN_DIR="$2" WSRC="$3"
  local OUT="$LOGS/diag_policy_side_${TAG}_${NAME}_${SEED_TAG}.json"
  [ -f "$RUN_DIR/params_${EPOCH}.pkl" ] || { echo "no params_${EPOCH}.pkl in $RUN_DIR" >&2; exit 1; }
  echo "=== $NAME: $RUN_DIR @ $EPOCH  w_source=$WSRC  out=$OUT"
  # shellcheck disable=SC2086
  srun .venv/bin/python tools/diag_measure_vs_scalar_q.py agent=psmflow \
    env_name="$ENV_NAME" \
    agent.flow_ckpt_path="$FLOW_DIR" agent.flow_ckpt_epoch=500000 \
    agent.preimage_path="$PRE_NPZ" agent.use_point_preimage=true \
    restore_path="$RUN_DIR" restore_epoch="$EPOCH" \
    +na_path="$NA_DIR" +na_epoch="$NA_EPOCH" \
    +reward_raw_path="$REWARD_RAW" +reward_scaled_path="$REWARD_SCALED" \
    +w_source="$WSRC" report_out="$OUT" $EXTRA
}

run_one gpi      "$GPI_DIR"   file
run_one sec10    "$SEC10_DIR" infer
run_one sec10bc0 "$BC0_DIR"   infer
