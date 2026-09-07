#!/usr/bin/env bash
# Stage C. PSMFlow representation training: learns phi / psi against the FROZEN Stage-A
# behaviour flow, using precomputed noise preimages as the behavioural index measure
# (psmflow plan Tasks 3-6).
#
# Requires BOTH Stage-A outputs to exist:
#   FLOW_CKPT   Stage-A run dir (restore_agent glob convention), from
#               scripts/pretrain_behavior_flow.sh
#   FLOW_EPOCH  the epoch of the Stage-A ckpt to load, e.g. 500000
#   PREIMAGES   the npz from tools/precompute_preimages.py, computed from THAT ckpt
#               (main.py asserts the npz row count matches the dataset, which catches
#               the wrong-env / stale-file case)
#
# Usage: FLOW_CKPT=<dir> FLOW_EPOCH=<n> PREIMAGES=<npz> \
#          bash scripts/launch_psmflow.sh ENV [GPU] [STEPS] [WANDB_MODE]
#   SEEDS="0 1 2"  space-separated seeds; EXTRA= extra hydra overrides
#
# EXTRA is normally EMPTY. Since 2026-09-04 the config defaults ARE the primary algorithm --
# the paper-strict affine LatentFlowPSM (psi_form=affine policy_index=latent
# train_actor=false acting=gpi, use_point_preimage=true) -- so a bare launch runs it. Use
# EXTRA only for ablations, e.g. EXTRA="agent.psi_form=free" or
# EXTRA="agent.train_actor=true agent.acting=actor".
#
# NOTE (one seed per GPU): invoke once per GPU with SEEDS="0", SEEDS="1", … rather than
# letting several seeds share one device (HANDOFF infra lesson).
set -euo pipefail

ENV="${1:?env_name required, e.g. cube-single-play-singletask-v0}"
GPU="${2:-1}"
STEPS="${3:-500000}"
WMODE="${4:-online}"
: "${FLOW_CKPT:?FLOW_CKPT=<stage-A run dir glob> required}"
: "${FLOW_EPOCH:?FLOW_EPOCH=<stage-A ckpt epoch> required}"
: "${PREIMAGES:?PREIMAGES=<preimages.npz> required}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
EVAL_INT="${EVAL_INT:-50000}"
SAVE_INT="${SAVE_INT:-250000}"
MEM_FRAC="${MEM_FRAC:-0.30}"
EXTRA="${EXTRA:-}"
GROUP="${GROUP:-psmflow_${ENV%%-singletask*}_$(date +%Y%m%d_%H%M%S)}"

# Remember whether DISCOUNT came in from the caller's environment (even "") before the
# env_name pattern match below fills in a default -- ${VAR+set} is unset-safe under `set -u`.
DISCOUNT_USER_SET="${DISCOUNT+set}"
# This script takes a raw env_name rather than scripts/slurm/train_psmflow.sbatch's ENVKEY,
# so match the same four env families by substring instead of a case over a short key.
# docs/design/2026-09-06-antmaze-failure-tests.md H2: gamma=0.98's 50-step horizon is
# shorter than antmaze-medium's 200-400 step goal distance and the arm sits at the BC
# floor; the 51-cell, 3-seed ladder found gamma=0.99 (0.294 +/- 0.070) clearly beats 0.98
# (0.076) and beats 0.995, which collapses to ~0.01 past 300k. configs/agent/psmflow.yaml
# keeps discount=0.98 so existing checkpoints still restore -- this is a launcher-level
# default, not a config-wide one.
case "$ENV" in
  antmaze*) DISCOUNT_DEFAULT=0.99 ;;
  cube*|pointmaze*|scene*) DISCOUNT_DEFAULT=0.98 ;;
  *) DISCOUNT_DEFAULT=0.98 ;;
esac
# DISCOUNT (if the caller set it) wins over the per-env default above.
DISCOUNT="${DISCOUNT:-$DISCOUNT_DEFAULT}"
# Only append a hydra override when it would actually change the yaml default (0.98) or
# was explicitly requested, and skip it if EXTRA already sets agent.discount itself (avoids
# a duplicate-key hydra error).
DISCOUNT_OVERRIDE=""
if [ -n "$DISCOUNT_USER_SET" ] || [ "$DISCOUNT" != "0.98" ]; then
  case "$EXTRA" in
    *agent.discount=*) : ;;  # caller already overrides it via EXTRA; don't double-set
    *) DISCOUNT_OVERRIDE="agent.discount=$DISCOUNT" ;;
  esac
fi

REPO=/u/amsks/git/PSMFLows
# Where checkpoints, wandb and logs go. /var/local is a 24G node-local partition and it
# FILLED on 2026-08-31, killing six runs mid-training with OSError errno 28; /data-local is
# 3.5T. Override with STORE=/data-local/amsks/PSMFLows (the default stays /var/local so
# every earlier launch line reproduces).
LOCAL="${STORE:-/var/local/amsks}"
PY="$REPO/.venv/bin/python"     # system `python` is 2.7 on midi-01
mkdir -p "$LOCAL/exp" "$LOCAL/wandb" "$LOCAL/logs/$GROUP"

# Fail here rather than 40 seconds into JIT if the preimage file is missing/mistyped.
[ -f "$PREIMAGES" ] || { echo "no such preimage npz: $PREIMAGES" >&2; exit 1; }

cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRAC"
export WANDB_MODE="$WMODE"
export WANDB_DIR="$LOCAL/wandb"
export WANDB_ENTITY=amsks
export OGBENCH_DATASET_DIR="$LOCAL/ogbench"
export MUJOCO_GL=egl            # midi-01 has no DISPLAY; pointmaze builds a renderer

for seed in "${SEEDS[@]}"; do
  log="$LOCAL/logs/$GROUP/sd${seed}.log"
  echo "launching psmflow sd${seed} on $ENV -> $log (discount=$DISCOUNT default=$DISCOUNT_DEFAULT override='${DISCOUNT_OVERRIDE:-none}')"
  nohup "$PY" main.py \
    agent=psmflow \
    agent.flow_ckpt_path="$FLOW_CKPT" agent.flow_ckpt_epoch="$FLOW_EPOCH" \
    agent.preimage_path="$PREIMAGES" $DISCOUNT_OVERRIDE $EXTRA \
    env_name="$ENV" \
    offline_steps="$STEPS" online_steps=0 \
    log_interval=5000 eval_interval="$EVAL_INT" eval_episodes=50 \
    save_interval="$SAVE_INT" save_dir="$LOCAL/exp" \
    run_group="$GROUP" seed="$seed" \
    > "$log" 2>&1 &
  sleep 8   # stagger JIT compiles so they don't collide on startup
done

echo "wandb group: $GROUP"
echo "logs: $LOCAL/logs/$GROUP/"
wait
