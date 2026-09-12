#!/usr/bin/env bash
# 2026-09-08 Item 2: the first arm on this substrate that is actually DSRL-NA.
# docs/design/2026-09-08-critic-signal-and-dsrl-na.md 3.
#
# NOT ZERO-SHOT. `dsrl_na.qa` is trained by TD on cube's REAL reward. This arm answers one
# question -- can the frozen flow be steered at all on cube, when the critic is allowed to
# be reward-specific and to see raw actions -- and its number must never be quoted beside a
# zero-shot row.
#
# What was missing before: the 2026-09-08 audit found the repo's "DSRL-NA" had copied
# DSRL's ACTOR (tanh-Gaussian over latents, auto-alpha) and none of DSRL's CRITIC. Real
# DSRL-NA is a dual critic -- qa(s,a) by TD on real reward, qw(s,u) regressed onto
# qa(s, G(s,u)) at prior latents, actor climbs qw only. `agent.actor_mode=dsrl_sac`
# supplies the actor; `agent.dsrl_na.enabled=true` supplies the critic; `create` asserts
# you configured both.
#
# Unlike the 09-08 `dsrlfaithful` arm this does NOT need policy_index=task_vector: psi is
# not in the actor's gradient at all here, so the paper-strict affine defaults stay on
# (psi/phi keep training alongside and are simply unused).
#
#   PSM_REPO=... PSM_DATA=... bash scripts/slurm/launch_dsrl_na.sh [smoke|table]
set -euo pipefail
: "${PSM_REPO:?}"; : "${PSM_DATA:?}"
ENVKEY="${ENVKEY:-cube}"; FLOW="${FLOW:-cube-single-play}"
LOGDIR="$PSM_DATA/logs/slurm/dsrlna"; mkdir -p "$LOGDIR"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-500000}"
# Arm D1 (2026-09-09): the same arm with qa's reward replaced by its reconstruction through
# the frozen basis, r_hat = phi(s')^T w. One change from Item 2, isolating the first of the
# four differences between the reward-specific arm and the zero-shot one.
# Arm D2 (2026-09-09): the ZERO-SHOT version. Critics and actor take the task vector, and
# the reward IS phi(s')^T w at the FB-style w `sample_step_inputs` already draws, so no real
# reward enters training at any point. Note this makes the measure branch load-bearing for
# the first time in this family -- phi defines the reward, so training it is no longer inert.
REWARD_SOURCE="${REWARD_SOURCE:-real}"
# Arm D1b only: how often the held reward-readout w is refit from a fresh relabel batch.
REFIT_EVERY="${REFIT_EVERY:-10000}"
TASK_COND="${TASK_COND:-false}"
case "$REWARD_SOURCE" in
  real)        GROUP="${GROUP:-dsrlna_${ENVKEY}}" ;;
  phi_readout) GROUP="${GROUP:-dsrlna_d1_${ENVKEY}}" ;;
  # Arm D1b (2026-09-09): D1 refit w on each 256-row batch, so Q_A was trained on a reward
  # redrawn every step whose std was 13.6 against the real reward's 0.15. Here w is the
  # closed form on an eval_relabel_size batch, refit every REFIT_EVERY steps and held fixed
  # between, and r_hat is rescaled onto the real reward's std -- the readout as DEPLOYED.
  phi_readout_fixed) GROUP="${GROUP:-dsrlna_d1b_${ENVKEY}}" ;;
  synthetic_w) GROUP="${GROUP:-dsrlna_d2_${ENVKEY}}" ;;
  *) echo "unknown REWARD_SOURCE: $REWARD_SOURCE" >&2; exit 1 ;;
esac
# Never delay the running zero-shot campaign; D1/D2 answer a question that keeps.
# `|| true`: with no matching jobs queued the grep exits 1, and under `set -o pipefail`
# that aborts the whole launcher before it prints anything.
DEPS=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null \
       | grep -E "usignal_arm_b|armc_" | awk '{print $1}' | paste -sd: - || true)
DEP_ARG=""
if [ -n "$DEPS" ]; then DEP_ARG="--dependency=afterany:$DEPS"; fi

# EXTRA carries agent overrides ONLY; train_psmflow.sbatch supplies agent=, flow_ckpt_*,
# preimage_path, env_name, offline_steps, save_dir, run_group and seed.
#
# Every value below that differs from configs/agent/psmflow.yaml is set here explicitly, so
# the launched command IS the hyperparameter table. The dsrl_na sub-keys are already at the
# recipe's values in the yaml and are restated for the same reason.
ARM="\
agent.dsrl_na.enabled=true \
agent.dsrl_na.discount=0.99 \
agent.dsrl_na.tau=0.005 \
agent.dsrl_na.lr=3.0e-4 \
agent.dsrl_na.hidden_dim=2048 \
agent.dsrl_na.hidden_layers=3 \
agent.dsrl_na.layer_norm=true \
agent.dsrl_na.num_ensembles=2 \
agent.dsrl_na.inner_steps=10 \
agent.dsrl_na.n_latent=1 \
agent.actor_mode=dsrl_sac \
agent.train_actor=true \
agent.acting=actor \
agent.actor.bc_coeff=0.0 \
agent.actor.q_coeff=1.0 \
agent.actor.entropy=auto \
agent.actor.target_entropy=0.0 \
agent.actor.init_alpha=1.0 \
agent.actor.lr_alpha=3.0e-4 \
agent.actor.prior_init=true \
agent.actor.prior_init_std=0.3 \
agent.actor.layer_norm=true \
agent.actor.hidden_dim=2048 \
agent.actor.hidden_layers=3 \
agent.lr_actor=3.0e-4 \
agent.u_clip=1.5 \
agent.batch_size=256 \
agent.dsrl_na.reward_source=$REWARD_SOURCE \
agent.dsrl_na.reward_refit_every=$REFIT_EVERY \
agent.dsrl_na.task_conditioned=$TASK_COND"

if [ "${1:-}" = "table" ]; then
  echo "group=$GROUP env=$ENVKEY steps=$STEPS seeds='$SEEDS'"
  echo "$ARM" | tr ' ' '\n' | sed '/^$/d' | sed 's/^/  /'
  exit 0
fi

# The smoke is the exact code path at 200 steps: same overrides, same flow, same npz, one
# eval, no checkpoint. Read steps/s off its log before committing 3 x 500k.
if [ "${1:-}" = "smoke" ]; then
  sbatch --partition=kisski-inference --account=general --gres=gpu:1 \
    --cpus-per-task=8 --mem=64G --time=00:40:00 --job-name="dsrlna_smoke" \
    --output="$LOGDIR/%x-%j.out" --error="$LOGDIR/%x-%j.err" \
    --export=ALL,PSM_REPO="$PSM_REPO",PSM_DATA="$PSM_DATA",ENVKEY="$ENVKEY",\
PREIMAGES="$PSM_DATA/preimages/$FLOW.npz",GROUP="dsrlna_smoke",SEED=0,STEPS=200,\
EVAL_INT=200,EVAL_EPS=2,LOG_INT=50,SAVE_INT=1000000,EXTRA="$ARM" \
    "$PSM_REPO/scripts/slurm/train_psmflow.sbatch"
  exit 0
fi

# Measured on the 200-step smoke: ~15 steps/s, three times slower than the affine arm --
# the 10 unrolled inner qw steps on 3 x 2048 nets. 500k steps is ~9.5 h plus ten in-loop
# evals, hence 24 h rather than the usual 8-12.
for s in $SEEDS; do
  # shellcheck disable=SC2086  -- DEP_ARG is one optional flag or empty
  sbatch --partition=kisski-inference --account=general --gres=gpu:1 $DEP_ARG \
    --cpus-per-task=8 --mem=64G --time=24:00:00 --job-name="${GROUP}_sd${s}" \
    --output="$LOGDIR/%x-%j.out" --error="$LOGDIR/%x-%j.err" \
    --export=ALL,PSM_REPO="$PSM_REPO",PSM_DATA="$PSM_DATA",ENVKEY="$ENVKEY",\
PREIMAGES="$PSM_DATA/preimages/$FLOW.npz",GROUP="$GROUP",SEED="$s",STEPS="$STEPS",\
SAVE_INT=50000,EXTRA="$ARM" \
    "$PSM_REPO/scripts/slurm/train_psmflow.sbatch"
done
echo "runs -> $PSM_DATA/exp/PSMFLows/$GROUP"
echo "AFTER LAUNCH: re-read each run's flags.json and confirm dsrl_na.enabled, "
echo "actor.bc_coeff=0, u_clip=1.5, batch_size=256 landed."
