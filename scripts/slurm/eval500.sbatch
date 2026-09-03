#!/usr/bin/env bash
# One 500-episode eval per SLURM job -- the scheduler-side wrapper around
# scripts/eval500.sh, the same way train_psmflow.sbatch wraps main.py.
#
# eval500.sh's paths default to midi-01; here they are pointed at PSM_DATA, which on this
# cluster holds flow/<name>/ and preimages/<name>.npz (the layout hf_preimages.py pulls
# into). Report JSONs land in $PSM_DATA/logs, which is the --logs dir to hand
# tools/make_tables.py on this machine.
#
# Submit:
#   sbatch --partition=kisski-inference --account=general --time=02:00:00 \
#     --job-name=e500_<name> --output=$PSM_DATA/logs/%x-%j.out \
#     --export=ALL,MODE=latentrl,ENVKEY=cube,RUN_DIR=<run dir>,OUT=<basename>,\
#              SEED=0,EXTRA='agent.u_clip=1.0 agent.critic_input=latent' \
#     scripts/slurm/eval500.sbatch
#
# EXTRA is forwarded verbatim as trailing hydra overrides. For latentrl it MUST repeat
# agent.critic_input and agent.u_clip from the run's flags.json -- the actor is
# tanh * u_clip, so a missing u_clip=1.0 scales every action by 3x.
#SBATCH --job-name=e500
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

: "${PSM_REPO:?PSM_REPO=<path to the PSMFLows checkout> required}"
: "${PSM_DATA:?PSM_DATA=<scratch dir with flow/ and preimages/> required}"
: "${MODE:?MODE=psmflow|latentrl|bc required}"
: "${ENVKEY:?ENVKEY=cube|antmaze|pointmaze required}"
OUT="${OUT:?OUT=<report json basename> required}"
RUN_DIR="${RUN_DIR:--}"
# Eval seed. Unset by default, which is the protocol every recorded eval500 number used:
# scripts/reeval_ode_e2.sh never passes one, so cfg.seed stays at the config default (0) for
# every training seed and the 500 episode inits are paired across seeds.
SEED="${SEED:-}"
EXTRA="${EXTRA:-}"

case "$ENVKEY" in
  cube)      FLOW=cube-single-play ;;
  antmaze)   FLOW=antmaze-medium-navigate ;;
  pointmaze) FLOW=pointmaze-medium-navigate ;;
  *) echo "unknown ENVKEY: $ENVKEY" >&2; exit 1 ;;
esac

export PSM_REPO
export FLOW_DIR="$PSM_DATA/flow/$FLOW"
export PRE_NPZ="$PSM_DATA/preimages/$FLOW.npz"
export EVAL_LOGS="$PSM_DATA/logs"
export MEM_FRAC="${MEM_FRAC:-0.90}"
export GPU=0
# ogbench's own default; $PSM_DATA/ogbench does not exist on this cluster.
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}"

echo "=== eval500: mode=$MODE env=$ENVKEY seed=${SEED:-<config default>} out=$OUT"
echo "    run_dir=$RUN_DIR"
echo "    flow=$FLOW_DIR  preimages=$PRE_NPZ"
echo "    extra='$EXTRA'"

cd "$PSM_REPO"
# shellcheck disable=SC2086  -- EXTRA is a deliberate word-split list of hydra overrides
SEED_ARG=()
if [ -n "$SEED" ]; then SEED_ARG=(seed="$SEED"); fi
srun bash scripts/eval500.sh "$MODE" "$ENVKEY" "$RUN_DIR" "$OUT" ${SEED_ARG[@]+"${SEED_ARG[@]}"} $EXTRA
