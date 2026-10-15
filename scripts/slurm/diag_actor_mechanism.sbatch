#!/usr/bin/env bash
# tools/diag_actor_mechanism.py on one GPU: two psmgoal checkpoints (critic A, the actor-bootstrap
# run; critic B, the prior-bootstrap run), the H1/H2/H3 actor-mechanism measurements on the same
# sampled (s, g) pairs. The scheduler-side wrapper, mirroring eval_state_reward.sbatch on KISSKI.
#
# Submit:
#   sbatch --partition=kisski-inference --account=general --time=02:00:00 \
#     --job-name=am_<tag> --output=$PSM_DATA/logs/slurm/%x-%j.out \
#     --export=ALL,RUN_DIR=<run dir A>,RUN_DIR_B=<run dir B>,EPOCH=500000,TAG=<tag> \
#     scripts/slurm/diag_actor_mechanism.sbatch
#
#   RUN_DIR    critic A's run directory (flags.json and params_<EPOCH>.pkl) (required)
#   RUN_DIR_B  critic B's run directory (required)
#   EPOCH      checkpoint epoch, both critics (default 500000)
#   TAG        report basename: $PSM_DATA/logs/actor_mechanism/${TAG}.json (required)
#   SEED       sampling seed (default 0)
#   EXTRA      trailing hydra overrides, e.g. "+n_states=8 +n_draws=200" for a smoke; the agent
#              config comes from each run's own flags.json
#SBATCH --job-name=diag_actor_mechanism
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
: "${RUN_DIR:?RUN_DIR=<critic A run dir> required}"
: "${RUN_DIR_B:?RUN_DIR_B=<critic B run dir> required}"
: "${TAG:?TAG=<report basename> required}"
EPOCH="${EPOCH:-500000}"
SEED="${SEED:-0}"
EXTRA="${EXTRA:-}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRAC:-0.90}"
export MUJOCO_GL=egl
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}"
export CUDA_VISIBLE_DEVICES=0

FLOW_DIR="$PSM_DATA/flow/cube-single-play"
PRE_NPZ="$PSM_DATA/preimages/cube-single-play.npz"
OUTD="$PSM_DATA/logs/actor_mechanism"
mkdir -p "$OUTD"

[ -d "$FLOW_DIR" ] || { echo "no such flow dir: $FLOW_DIR" >&2; exit 1; }
[ -f "$PRE_NPZ" ] || { echo "no such preimage npz: $PRE_NPZ" >&2; exit 1; }
for d in "$RUN_DIR" "$RUN_DIR_B"; do
  [ -f "$d/params_${EPOCH}.pkl" ] || { echo "no such checkpoint: $d/params_${EPOCH}.pkl" >&2; exit 1; }
done

echo "=== diag_actor_mechanism: epoch=$EPOCH seed=$SEED"
echo "    A=$RUN_DIR"
echo "    B=$RUN_DIR_B"
echo "    flow=$FLOW_DIR preimages=$PRE_NPZ"
echo "    out=$OUTD/${TAG}.json extra='$EXTRA'"

cd "$PSM_REPO"
# shellcheck disable=SC2086  -- EXTRA is a deliberate word-split list of hydra overrides
srun "$PSM_REPO/.venv/bin/python" tools/diag_actor_mechanism.py agent=psmgoal \
  env_name=cube-single-play-singletask-task1-v0 \
  agent.flow_ckpt_path="$FLOW_DIR" agent.flow_ckpt_epoch=500000 \
  agent.preimage_path="$PRE_NPZ" agent.use_point_preimage=true \
  restore_path="$RUN_DIR" +restore_path_b="$RUN_DIR_B" restore_epoch="$EPOCH" \
  seed="$SEED" report_out="$OUTD/${TAG}.json" $EXTRA
