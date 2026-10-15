#!/usr/bin/env bash
# tools/diag_time_to_success.py on one GPU: one psmgoal checkpoint (critic A), optionally a second
# (critic B), every readout and all five cube tasks in a single process. Same partition/account/
# paths as eval_state_reward.sbatch on KISSKI.
#
# Submit:
#   sbatch --partition=kisski-inference --account=general --time=02:00:00 \
#     --job-name=tts_<tag> --output=$PSM_DATA/logs/slurm/%x-%j.out \
#     --export=ALL,RUN_DIR=<critic A run>,RUN_DIR_B=<critic B run>,EPOCH=500000,READOUTS="bon64 bon8 bc actor",EPISODES=100,TAG=<tag> \
#     scripts/slurm/diag_time_to_success.sbatch
#
#   RUN_DIR    the run directory holding flags.json and params_<EPOCH>.pkl (required, critic A)
#   RUN_DIR_B  a second run directory (critic B); adds the bon64_B readout when set (optional)
#   EPOCH      checkpoint epoch (default 500000)
#   READOUTS   space-separated readout names: bon64 bon8 bc actor [bon64_B] (default "bon64 bon8 bc actor")
#   EPISODES   episodes per task (default 100)
#   TAG        report basename: $PSM_DATA/logs/time_to_success/${TAG}.json (required)
#   TASKS      tasks evaluated (default "1 2 3 4 5")
#   SEED       eval seed (default 0)
#   EXTRA      trailing hydra overrides, normally EMPTY: the agent config comes from the run's own
#              flags.json; pass agent.* keys only to evaluate deliberately off its config
#SBATCH --job-name=diag_time_to_success
# SLURM does NOT expand environment variables in #SBATCH directives: this is the literal
# $PSM_DATA/logs/slurm on KISSKI. Launchers override both flags.
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
set -euo pipefail

PSM_REPO="${PSM_REPO:-/mnt/home/amohan/git/Austin/PSMFLows}"
export PSM_DATA="${PSM_DATA:-/mnt/home/amohan/psm-data}"
: "${RUN_DIR:?RUN_DIR=<run dir with flags.json and params_<EPOCH>.pkl> required}"
: "${TAG:?TAG=<report basename> required}"
RUN_DIR_B="${RUN_DIR_B:-}"
EPOCH="${EPOCH:-500000}"
READOUTS="${READOUTS:-bon64 bon8 bc actor}"
EPISODES="${EPISODES:-100}"
TASKS="${TASKS:-1 2 3 4 5}"
SEED="${SEED:-0}"
EXTRA="${EXTRA:-}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRAC:-0.90}"
export MUJOCO_GL=egl
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}"
export CUDA_VISIBLE_DEVICES=0

FLOW_DIR="$PSM_DATA/flow/cube-single-play"
PRE_NPZ="$PSM_DATA/preimages/cube-single-play.npz"
OUTD="$PSM_DATA/logs/time_to_success"
mkdir -p "$OUTD"

[ -d "$FLOW_DIR" ] || { echo "no such flow dir: $FLOW_DIR" >&2; exit 1; }
[ -f "$PRE_NPZ" ] || { echo "no such preimage npz: $PRE_NPZ" >&2; exit 1; }
[ -f "$RUN_DIR/params_${EPOCH}.pkl" ] || { echo "no such checkpoint: $RUN_DIR/params_${EPOCH}.pkl" >&2; exit 1; }
B_ARG=""
if [ -n "$RUN_DIR_B" ]; then
  [ -f "$RUN_DIR_B/params_${EPOCH}.pkl" ] || { echo "no such checkpoint: $RUN_DIR_B/params_${EPOCH}.pkl" >&2; exit 1; }
  B_ARG="+restore_path_b=$RUN_DIR_B"
fi

echo "=== diag_time_to_success: readouts='$READOUTS' episodes/task=$EPISODES tasks='$TASKS' seed=$SEED"
echo "    run_dir=$RUN_DIR run_dir_b='$RUN_DIR_B' epoch=$EPOCH"
echo "    flow=$FLOW_DIR preimages=$PRE_NPZ"
echo "    out=$OUTD/${TAG}.json extra='$EXTRA'"

cd "$PSM_REPO"
# shellcheck disable=SC2086  -- EXTRA / B_ARG are deliberate word-split lists of hydra overrides
srun "$PSM_REPO/.venv/bin/python" tools/diag_time_to_success.py agent=psmgoal \
  env_name=cube-single-play \
  agent.flow_ckpt_path="$FLOW_DIR" agent.flow_ckpt_epoch=500000 \
  agent.preimage_path="$PRE_NPZ" agent.use_point_preimage=true \
  restore_path="$RUN_DIR" restore_epoch="$EPOCH" $B_ARG \
  +readouts="$READOUTS" +tasks="$TASKS" \
  eval_episodes="$EPISODES" seed="$SEED" \
  report_out="$OUTD/${TAG}.json" $EXTRA
