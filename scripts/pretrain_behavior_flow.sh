#!/usr/bin/env bash
# Stage A. Pretrain the reward-free BEHAVIOUR FLOW G_theta(s,u) that PSMFlows freezes and
# indexes policies by (psmflow plan Task 1). This is FQL with bc_only=true: no critic, no
# Q term, rewards/masks never read — what remains is the conditional-flow-matching
# objective plus its one-step distillation.
#
# The checkpoint this writes is the input to the preimage pipeline
# (tools/precompute_preimages.py) and to diagnostics D1/D3
# (tools/validate_flow_fidelity.py, tools/validate_flow_inversion.py).
#
# This is the plan's `launch_flowbc.sh` under the name Task 1 already established;
# tools/precompute_preimages.py and tools/validate_flow_inversion.py cite this path.
#
# Usage: bash scripts/pretrain_behavior_flow.sh [GPU] [STEPS] [ENV]
#   SEEDS="0 1 2"  space-separated seeds (SEED= still honoured for one)
#   FLOW_STEPS=    ODE discretization used in training (see below)
#   SAVE_INT=      checkpoint interval; EXTRA= extra hydra overrides
#
# NOTE (one seed per GPU): the multi-seed loop exists for convenience, but the HANDOFF
# infra lesson is to invoke this once per GPU with SEEDS="0", SEEDS="1", … rather than
# letting several seeds share one device.
set -euo pipefail

GPU="${1:-1}"
STEPS="${2:-500000}"
ENV="${3:-cube-single-play-singletask-v0}"
read -r -a SEEDS <<< "${SEEDS:-${SEED:-0}}"
# The preimage inversion requires the forward and inverse maps to share a discretization
# at >= 100 steps (tools/precompute_preimages.py asserts it), so Stage A trains at 100.
# The landed cube ckpt (bcflow_cube_single_20260726_135032) predates this and is at the
# FQL default of 10; it is inverted by overriding agent.flow_steps=100 at precompute time.
FLOW_STEPS="${FLOW_STEPS:-100}"
SAVE_INT="${SAVE_INT:-250000}"
MEM_FRAC="${MEM_FRAC:-0.25}"
EXTRA="${EXTRA:-}"
GROUP="${GROUP:-bcflow_$(echo "$ENV" | cut -d- -f1-3)_$(date +%Y%m%d_%H%M%S)}"

REPO=/u/amsks/git/PSMFLows
LOCAL=/var/local/amsks
PY="$REPO/.venv/bin/python"     # system `python` is 2.7 on midi-01
mkdir -p "$LOCAL/exp" "$LOCAL/wandb" "$LOCAL/logs/$GROUP"

cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRAC"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="$LOCAL/wandb"
export WANDB_ENTITY=amsks
export OGBENCH_DATASET_DIR="$LOCAL/ogbench"

# eval_interval=0 disables eval: task success is meaningless for a reward-free BC flow.
for seed in "${SEEDS[@]}"; do
  log="$LOCAL/logs/$GROUP/bcflow_sd${seed}.log"
  echo "pretraining behaviour flow on $ENV, GPU $GPU, $STEPS steps, flow_steps=$FLOW_STEPS -> $log"
  nohup "$PY" main.py \
    agent=fql agent.bc_only=true agent.flow_steps="$FLOW_STEPS" $EXTRA \
    env_name="$ENV" \
    offline_steps="$STEPS" online_steps=0 \
    log_interval=5000 eval_interval=0 \
    save_interval="$SAVE_INT" save_dir="$LOCAL/exp" \
    run_group="$GROUP" seed="$seed" \
    > "$log" 2>&1 &
  sleep 8   # stagger JIT compiles so they don't collide on startup
  echo "  ckpt -> $LOCAL/exp/PSMFLows/$GROUP/sd$(printf '%03d' "$seed")_*/params_${STEPS}.pkl"
done

echo "wandb group: $GROUP"
echo "logs: $LOCAL/logs/$GROUP/"
wait
