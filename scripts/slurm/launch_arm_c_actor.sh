#!/usr/bin/env bash
# 2026-09-09: Arm C's value, deployed by a GRADIENT ACTOR instead of the best-of-64 argmax.
#
# The question. Arm C fits the measure head at 4 latents per transition drawn from the
# shrunken EM posterior, and its eval500 ladder is monotone in all three seeds, pooled 0.503
# against the control's 0.442. If that extra fitting gave psi real SLOPE over u -- rather
# than just a better ordering at 64 sampled points -- then an actor that climbs psi^T w by
# gradient should work on this value where it did not on the control's. If it does not, the
# slope is not there and best-of-64 stays the deployment for this value.
#
# WHAT THIS ARM IS NOT. The 2026-09-08 "faithful DSRL" arm scored 0.136/0.141 and is the
# obvious comparator, but it is NOT one change away from this. Read off
# dsrlfaithful_cube_uc15/sd000's own flags.json, it ran:
#
#     psi_form=free          policy_index=task_vector   index_agg=max
#     batch_size=1024        lr_actor=1e-4              actor 512x2, no LayerNorm
#
# against Arm C's paper-strict affine value (psi_form=affine, policy_index=latent) at
# batch_size=256. Those are different VALUE FUNCTIONS, not different acting rules, so
# "match the faithful arm exactly and change only the latents" and "keep Arm C exactly"
# cannot both hold. This script keeps ARM C exactly and changes only the deployment,
# because the question is about Arm C's value. The clean comparator is therefore Arm C's
# own best-of-64 ladder (0.503 pooled, monotone 9/9); 0.136/0.141 is indicative only and
# must not be quoted as a one-change control.
#
# The actor settings are the faithful arm's INTENT (bc_coeff=0, auto entropy at target 0,
# mode action at eval) at the DSRL-NA arm's widths, which is the configuration that actually
# worked on this substrate.
#
# WHAT IS DELIBERATELY *NOT* SET HERE. The DSRL-NA recipe also carries u_clip=1.5 and
# batch_size=256, and an earlier version of this script copied both. That was wrong: Arm C
# runs at u_clip=3.0 and batch_size=1024 (read off armc_cube/sd000's flags.json). u_clip is
# the latent box the measure loss clips u_data into and the box the value was fitted and
# evaluated over -- narrowing it changes Arm C's VALUE, not its deployment -- and the batch
# size is an optimisation setting the ladder was produced at. Both are therefore left at the
# yaml/Arm C defaults so this arm differs from armc_cube in the deployment alone.
#
# The one genuinely new degree of freedom is `lr_actor`: Arm C never trained an actor, so
# its 1e-4 was inert. 3e-4 is the DSRL-NA value, which is the rate that worked for a latent
# actor on this substrate.
#
#   PSM_REPO=... PSM_DATA=... bash scripts/slurm/launch_arm_c_actor.sh [table|smoke]
set -euo pipefail
: "${PSM_REPO:?}"; : "${PSM_DATA:?}"
ENVKEY="${ENVKEY:-cube}"; NPZ_NAME="${NPZ_NAME:-cube-single-play}"
LOGDIR="$PSM_DATA/logs/slurm/armcactor"; mkdir -p "$LOGDIR"
SEEDS="${SEEDS:-0 1 2}"; STEPS="${STEPS:-500000}"
GROUP="${GROUP:-armc_actor_${ENVKEY}}"
NPZ="$PSM_DATA/preimages/${NPZ_NAME}.npz"

# Every value that differs from configs/agent/psmflow.yaml is set here, so the launched
# command IS the hyperparameter table. First block: Arm C, byte-identical to launch_arm_c.sh.
# Second block: the gradient actor on the measure readout. dsrl_na stays OFF -- the actor
# climbs psi^T w at the random task vector each row draws, as the faithful arm did.
ARM="\
agent.measure_u_samples=4 \
agent.measure_u_source=mixture \
agent.measure_u_mixture_shrink=0.5 \
agent.train_actor=true \
agent.acting=actor \
agent.actor_mode=dsrl_sac \
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
agent.actor.index_panel=0 \
agent.lr_actor=3.0e-4"

if [ "${1:-}" = "table" ]; then
  echo "group=$GROUP env=$ENVKEY steps=$STEPS seeds='$SEEDS'"
  echo "  npz  $NPZ"
  echo "  psi_form=affine policy_index=latent  (Arm C's value, UNCHANGED -- NOT the"
  echo "    faithful arm's psi_form=free / policy_index=task_vector; see the header)"
  echo "$ARM" | tr ' ' '\n' | sed '/^$/d' | sed 's/^/  /'
  exit 0
fi

[ -f "$NPZ" ] || { echo "no such preimage npz: $NPZ" >&2; exit 1; }

if [ "${1:-}" = "smoke" ]; then
  sbatch --partition=kisski-inference --account=general --gres=gpu:1 \
    --cpus-per-task=8 --mem=64G --time=00:40:00 --job-name="armcactor_smoke" \
    --output="$LOGDIR/%x-%j.out" --error="$LOGDIR/%x-%j.err" \
    --export=ALL,PSM_REPO="$PSM_REPO",PSM_DATA="$PSM_DATA",ENVKEY="$ENVKEY",\
PREIMAGES="$NPZ",GROUP="armcactor_smoke",SEED=0,STEPS=200,\
EVAL_INT=200,EVAL_EPS=2,LOG_INT=50,SAVE_INT=1000000,EXTRA="$ARM" \
    "$PSM_REPO/scripts/slurm/train_psmflow.sbatch"
  exit 0
fi

for s in $SEEDS; do
  sbatch --partition=kisski-inference --account=general --gres=gpu:1 \
    --cpus-per-task=8 --mem=64G --time=12:00:00 --job-name="${GROUP}_sd${s}" \
    --output="$LOGDIR/%x-%j.out" --error="$LOGDIR/%x-%j.err" \
    --export=ALL,PSM_REPO="$PSM_REPO",PSM_DATA="$PSM_DATA",ENVKEY="$ENVKEY",\
PREIMAGES="$NPZ",GROUP="$GROUP",SEED="$s",STEPS="$STEPS",SAVE_INT=50000,EXTRA="$ARM" \
    "$PSM_REPO/scripts/slurm/train_psmflow.sbatch"
done
echo "runs -> $PSM_DATA/exp/PSMFLows/$GROUP"
echo "AFTER LAUNCH: re-read each flags.json for psi_form=affine, policy_index=latent,"
echo "measure_u_source=mixture, measure_u_mixture_shrink=0.5, acting=actor, bc_coeff=0."
