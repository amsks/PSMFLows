#!/usr/bin/env bash
# tools/diag_actor_shift_direction.py on one GPU: where the flowbc actor's shift points, on one
# psmgoal checkpoint (critic A). Same partition/account/paths as diag_time_to_success.sbatch.
#
# Submit:
#   sbatch --partition=kisski-inference --account=general --time=01:00:00 \
#     --job-name=shift_<tag> --output=$PSM_DATA/logs/slurm/%x-%j.out \
#     --export=ALL,RUN_DIR=<critic A run>,EPOCH=500000,TAG=<tag> \
#     scripts/slurm/diag_actor_shift_direction.sbatch
#
#   RUN_DIR    the run directory holding flags.json and params_<EPOCH>.pkl (required, flowbc actor)
#   EPOCH      checkpoint epoch (default 500000)
#   TAG        report basename: $PSM_DATA/logs/actor_shift/${TAG}.json (required)
#   N_STATES   sampled (s, g) rows (default 512); N_DRAWS prior draws per state (default 2000)
#   SEED       sampling seed (default 0)
#   EXTRA      trailing hydra overrides, normally EMPTY: the agent config comes from the run's own
#              flags.json; pass agent.* keys only to evaluate deliberately off its config
#SBATCH --job-name=diag_actor_shift_direction
# SLURM does NOT expand environment variables in #SBATCH directives: this is the literal
# $PSM_DATA/logs/slurm on KISSKI. Launchers override both flags.
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
set -euo pipefail

PSM_REPO="${PSM_REPO:-/mnt/home/amohan/git/Austin/PSMFLows}"
export PSM_DATA="${PSM_DATA:-/mnt/home/amohan/psm-data}"
: "${RUN_DIR:?RUN_DIR=<run dir with flags.json and params_<EPOCH>.pkl> required}"
: "${TAG:?TAG=<report basename> required}"
EPOCH="${EPOCH:-500000}"
N_STATES="${N_STATES:-512}"
N_DRAWS="${N_DRAWS:-2000}"
SEED="${SEED:-0}"
EXTRA="${EXTRA:-}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRAC:-0.90}"
export MUJOCO_GL=egl
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}"
export CUDA_VISIBLE_DEVICES=0

FLOW_DIR="$PSM_DATA/flow/cube-single-play"
PRE_NPZ="$PSM_DATA/preimages/cube-single-play.npz"
OUTD="$PSM_DATA/logs/actor_shift"
mkdir -p "$OUTD"

[ -d "$FLOW_DIR" ] || { echo "no such flow dir: $FLOW_DIR" >&2; exit 1; }
[ -f "$PRE_NPZ" ] || { echo "no such preimage npz: $PRE_NPZ" >&2; exit 1; }
[ -f "$RUN_DIR/params_${EPOCH}.pkl" ] || { echo "no such checkpoint: $RUN_DIR/params_${EPOCH}.pkl" >&2; exit 1; }

echo "=== diag_actor_shift_direction: n_states=$N_STATES n_draws=$N_DRAWS seed=$SEED"
echo "    run_dir=$RUN_DIR epoch=$EPOCH"
echo "    flow=$FLOW_DIR preimages=$PRE_NPZ"
echo "    out=$OUTD/${TAG}.json extra='$EXTRA'"

cd "$PSM_REPO"
# shellcheck disable=SC2086  -- EXTRA is a deliberate word-split list of hydra overrides
srun "$PSM_REPO/.venv/bin/python" tools/diag_actor_shift_direction.py agent=psmgoal \
  env_name=cube-single-play-singletask-task1-v0 \
  agent.flow_ckpt_path="$FLOW_DIR" agent.flow_ckpt_epoch=500000 \
  agent.preimage_path="$PRE_NPZ" agent.use_point_preimage=true \
  restore_path="$RUN_DIR" restore_epoch="$EPOCH" \
  +n_states="$N_STATES" +n_draws="$N_DRAWS" seed="$SEED" \
  report_out="$OUTD/${TAG}.json" $EXTRA
