#!/usr/bin/env bash
# The policy-family diversity probe (tools/diag_policy_family_diversity.py) in ONE job:
# noise_index / bc / random_action / tilted_bc(_strong) / noise_index_consistent, K members
# x N episodes each from the same N initial seeds through the affine run's frozen flow, plus
# a DSRL-NA latent actor (and optionally a second training seed of it) as the goal-directed
# reference. Eval only.
#
# Submit:
#   sbatch --partition=kisski-inference --account=general --time=01:30:00 \
#     --job-name=pfd_cube --output=$PSM_DATA/logs/%x-%j.out \
#     --export=ALL,PSM_REPO=<checkout>,RUN_DIR=<affine_strict_cube run dir>,\
#              NA_DIR=<dsrlna_cube run dir>[,NA_DIR2=<second seed's run dir>],TAG=cube \
#     scripts/slurm/diag_policy_family_diversity.sbatch
#
# Writes $PSM_DATA/logs/diag_policy_family_diversity_<TAG>.json plus a .npz sidecar with
# every member's visited-state and action clouds and the pairwise MMD^2 matrices.
#SBATCH --job-name=pfd
# SLURM does NOT expand environment variables in #SBATCH directives, so this is the
# literal $PSM_DATA/logs/slurm on KISSKI rather than the variable.
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

: "${PSM_REPO:?PSM_REPO=<path to the PSMFLows checkout> required}"
: "${PSM_DATA:?PSM_DATA=<scratch dir with flow/ and preimages/> required}"
: "${RUN_DIR:?RUN_DIR=<affine run dir with flags.json and params_*.pkl> required}"
: "${NA_DIR:?NA_DIR=<DSRL-NA run dir with flags.json and params_*.pkl> required}"
EPOCH="${EPOCH:-500000}"
NA_EPOCH="${NA_EPOCH:-500000}"
NA_DIR2="${NA_DIR2:-}"
TAG="${TAG:-cube}"
ENV_NAME="${ENV_NAME:-cube-single-play-singletask-task2-v0}"
FLOW="${FLOW:-cube-single-play}"
N_MEMBERS="${N_MEMBERS:-16}"
N_EP="${N_EP:-20}"
N_COH="${N_COH:-2000}"
EXTRA="${EXTRA:-}"
if [ -n "$NA_DIR2" ]; then
  EXTRA="$EXTRA +na_path2=$NA_DIR2"
fi

FLOW_DIR="$PSM_DATA/flow/$FLOW"
PRE_NPZ="$PSM_DATA/preimages/$FLOW.npz"
LOGS="$PSM_DATA/logs"
OUT="$LOGS/diag_policy_family_diversity_${TAG}.json"
[ -d "$FLOW_DIR" ] || { echo "no such flow dir: $FLOW_DIR" >&2; exit 1; }
[ -f "$PRE_NPZ" ]  || { echo "no such preimage npz: $PRE_NPZ" >&2; exit 1; }
[ -f "$RUN_DIR/params_${EPOCH}.pkl" ] || { echo "no params_${EPOCH}.pkl in $RUN_DIR" >&2; exit 1; }
[ -f "$NA_DIR/params_${NA_EPOCH}.pkl" ] || { echo "no params_${NA_EPOCH}.pkl in $NA_DIR" >&2; exit 1; }
mkdir -p "$LOGS"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRAC:-0.90}"
export MUJOCO_GL=egl
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}"

echo "=== diag_policy_family_diversity: env=$ENV_NAME tag=$TAG"
echo "    affine=$RUN_DIR @ $EPOCH"
echo "    dsrl-na=$NA_DIR @ $NA_EPOCH  second seed: ${NA_DIR2:-none}"
echo "    K=$N_MEMBERS N=$N_EP coherence states=$N_COH"
echo "    out=$OUT"

cd "$PSM_REPO"
# shellcheck disable=SC2086  -- EXTRA is a deliberate word-split list of hydra overrides
srun .venv/bin/python tools/diag_policy_family_diversity.py agent=psmflow \
  env_name="$ENV_NAME" \
  agent.flow_ckpt_path="$FLOW_DIR" agent.flow_ckpt_epoch=500000 \
  agent.preimage_path="$PRE_NPZ" agent.use_point_preimage=true \
  restore_path="$RUN_DIR" restore_epoch="$EPOCH" \
  +na_path="$NA_DIR" +na_epoch="$NA_EPOCH" \
  +n_members="$N_MEMBERS" +n_episodes="$N_EP" +n_coherence_states="$N_COH" \
  report_out="$OUT" $EXTRA
