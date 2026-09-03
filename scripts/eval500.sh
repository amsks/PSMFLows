#!/usr/bin/env bash
# 500-episode evaluation of one saved checkpoint (WP0 of the ICLR figure plan).
#
# Every number destined for the paper comes from here or from a multi-seed aggregate of
# these JSONs -- never from the 50-episode in-loop evals, whose 95% CI is about +/-0.115.
#
# Usage:
#   GPU=0 bash scripts/eval500.sh psmflow  cube  <run_dir> <out_name> [extra hydra args]
#   GPU=0 bash scripts/eval500.sh latentrl cube  <run_dir> <out_name> [extra hydra args]
#   GPU=0 bash scripts/eval500.sh bc       cube  -          <out_name>
#
# latentrl acts exactly as psmflow does (actor latent -> frozen one-step decode) but is a
# DIFFERENT hydra group, and restoring a latentrl checkpoint into agent=psmflow would fail
# or, worse, half-succeed. Two flags must be REPEATED here or the eval silently measures a
# different policy than the one trained:
#   agent.critic_input=latent -- the critic is built on latents; restoring without it can
#     only half-match, and the run's own flags.json is the source of truth;
#   agent.u_clip=<the training value> -- the actor is tanh * u_clip, so evaluating a
#     u_clip=1.0 checkpoint under the config default 3.0 scales every action by 3x.
#
# The second argument selects the env/flow/preimage triple; `bc` evaluates the frozen
# Stage-A flow with per-step prior latents (agent=fql agent.bc_only=true), which is the
# control every Stage-C number must be quoted beside.
set -euo pipefail

MODE="${1:?psmflow|latentrl|bc}"
ENVKEY="${2:?cube|antmaze|pointmaze}"
RUN_DIR="${3:?run dir, or - for bc}"
OUT="${4:?output json basename}"
shift 4
EXTRA=("$@")

case "$MODE" in psmflow|latentrl|bc) ;; *) echo "unknown mode: $MODE" >&2; exit 1 ;; esac

# Paths default to midi-01, where every earlier eval500 JSON was produced. On another
# machine (KISSKI/SLURM) override them in the environment rather than editing this file:
#   PSM_REPO   checkout root            EVAL_LOGS  where the report JSON is written
#   EXP_ROOT   experiment root          FLOW_DIR   frozen Stage-A ckpt dir (overrides ENVKEY)
#   PRE_NPZ    preimage npz (overrides ENVKEY)
REPO="${PSM_REPO:-/u/amsks/git/PSMFLows}"
PY="${PY:-$REPO/.venv/bin/python}"
LOGS="${EVAL_LOGS:-/data-local/amsks/PSMFLows/logs}"
EXP="${EXP_ROOT:-/var/local/amsks/exp/PSMFLows}"

case "$ENVKEY" in
  cube)
    ENV_NAME=cube-single-play-singletask-v0
    FLOW="${FLOW_DIR:-$EXP/bcflow_cube_single_20260726_135032/sd000_20260726_135037}"
    PRE="${PRE_NPZ:-/data-local/amsks/PSMFLows/preimages_cube_single_a20_n200.npz}" ;;
  antmaze)
    ENV_NAME=antmaze-medium-navigate-singletask-v0
    FLOW="${FLOW_DIR:-$EXP/bcflow_antmaze-medium-navigate_20260805_014546/sd000_20260805_014548}"
    PRE="${PRE_NPZ:-/data-local/amsks/PSMFLows/preimages_antmaze_medium_a20_n200.npz}" ;;
  pointmaze)
    ENV_NAME=pointmaze-medium-navigate-singletask-task1-v0
    FLOW="${FLOW_DIR:-$EXP/bcflow_pointmaze-medium-navigate_20260729_142219/sd000_20260729_142225}"
    PRE="${PRE_NPZ:-/data-local/amsks/PSMFLows/preimages_pointmaze_medium_a20_n200.npz}" ;;
  *) echo "unknown env key: $ENVKEY" >&2; exit 1 ;;
esac

[ -d "$FLOW" ] || { echo "no such flow dir: $FLOW" >&2; exit 1; }
if [ "$MODE" != "bc" ]; then
  [ -f "$PRE" ] || { echo "no such preimage npz: $PRE" >&2; exit 1; }
fi

cd "$REPO"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRAC:-0.30}"
export MUJOCO_GL=egl
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-/var/local/amsks/ogbench}"
mkdir -p "$LOGS"

if [ "$MODE" = "bc" ]; then
  # The frozen flow acting alone: sample_actions draws a fresh N(0, I) latent per step
  # and decodes it. No z inference, no preimages.
  "$PY" tools/eval_checkpoint.py agent=fql agent.bc_only=true \
    env_name="$ENV_NAME" restore_path="$FLOW" restore_epoch=500000 \
    eval_episodes=500 report_out="$LOGS/$OUT.json" "${EXTRA[@]}"
else
  "$PY" tools/eval_checkpoint.py agent="$MODE" \
    env_name="$ENV_NAME" \
    agent.flow_ckpt_path="$FLOW" agent.flow_ckpt_epoch=500000 \
    agent.preimage_path="$PRE" agent.use_point_preimage=true \
    restore_path="$RUN_DIR" restore_epoch=500000 \
    eval_episodes=500 report_out="$LOGS/$OUT.json" "${EXTRA[@]}"
fi
