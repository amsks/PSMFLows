#!/usr/bin/env bash
# tools/eval_state_reward.py on one GPU: one psmgoal checkpoint, one named state reward, one
# readout, all five cube tasks in a single process. The scheduler-side wrapper, the way
# eval500.sbatch wraps eval500.sh; same partition/account/paths as eval500.sbatch on KISSKI.
#
# Submit:
#   sbatch --partition=kisski-inference --account=general --time=03:00:00 \
#     --job-name=sr_<tag> --output=$PSM_DATA/logs/slurm/%x-%j.out \
#     --export=ALL,RUN_DIR=<run dir>,EPOCH=500000,REWARD=cube_height,READOUT=reward_gpi,EPISODES=50,TAG=<tag> \
#     scripts/slurm/eval_state_reward.sbatch
#
#   RUN_DIR   the run directory holding flags.json and params_<EPOCH>.pkl (required)
#   EPOCH     checkpoint epoch (default 500000)
#   REWARD    a STATE_REWARDS name: cube_height | cube_x | cube_y | effector_height | neg_cube_height
#   READOUT   reward_gpi | hgoal_topk | bc
#   EPISODES  episodes per task (default 50)
#   TAG       report basename: $PSM_DATA/logs/state_reward/${TAG}.json (required)
#   TASKS     tasks evaluated (default "1 2 3 4 5")
#   SEED      eval seed (default 0)
#   EXTRA     trailing hydra overrides, normally EMPTY: the agent config comes from the run's own
#             flags.json; pass agent.* keys only to evaluate deliberately off its config
#SBATCH --job-name=eval_state_reward
# SLURM does NOT expand environment variables in #SBATCH directives: this is the literal
# $PSM_DATA/logs/slurm on KISSKI. Launchers override both flags.
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
set -euo pipefail

PSM_REPO="${PSM_REPO:-/mnt/home/amohan/git/Austin/PSMFLows}"
export PSM_DATA="${PSM_DATA:-/mnt/home/amohan/psm-data}"
: "${RUN_DIR:?RUN_DIR=<run dir with flags.json and params_<EPOCH>.pkl> required}"
: "${TAG:?TAG=<report basename> required}"
EPOCH="${EPOCH:-500000}"
REWARD="${REWARD:-cube_height}"
READOUT="${READOUT:-reward_gpi}"
EPISODES="${EPISODES:-50}"
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
OUTD="$PSM_DATA/logs/state_reward"
mkdir -p "$OUTD"

[ -d "$FLOW_DIR" ] || { echo "no such flow dir: $FLOW_DIR" >&2; exit 1; }
[ -f "$PRE_NPZ" ] || { echo "no such preimage npz: $PRE_NPZ" >&2; exit 1; }
[ -f "$RUN_DIR/params_${EPOCH}.pkl" ] || { echo "no such checkpoint: $RUN_DIR/params_${EPOCH}.pkl" >&2; exit 1; }

echo "=== eval_state_reward: reward=$REWARD readout=$READOUT episodes/task=$EPISODES tasks='$TASKS' seed=$SEED"
echo "    run_dir=$RUN_DIR epoch=$EPOCH"
echo "    flow=$FLOW_DIR preimages=$PRE_NPZ"
echo "    out=$OUTD/${TAG}.json extra='$EXTRA'"

cd "$PSM_REPO"
# shellcheck disable=SC2086  -- EXTRA is a deliberate word-split list of hydra overrides
srun "$PSM_REPO/.venv/bin/python" tools/eval_state_reward.py agent=psmgoal \
  env_name=cube-single-play \
  agent.flow_ckpt_path="$FLOW_DIR" agent.flow_ckpt_epoch=500000 \
  agent.preimage_path="$PRE_NPZ" agent.use_point_preimage=true \
  restore_path="$RUN_DIR" restore_epoch="$EPOCH" \
  +reward="$REWARD" +readout="$READOUT" +tasks="$TASKS" \
  eval_episodes="$EPISODES" seed="$SEED" \
  report_out="$OUTD/${TAG}.json" $EXTRA
