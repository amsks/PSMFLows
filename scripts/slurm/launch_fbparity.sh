#!/usr/bin/env bash
# 2026-09-10 "fbparity": the psmflow affine measure with TD-JEPA/FB's OPTIMISATION recipe.
#
# Why. Every arm that optimises the fitted reward hard lands at or below the BC control
# (gradient actor 0.000, D2 0.004, D1b 0.011, BC 0.072) while best-of-64 argmax reaches
# 0.38-0.50 and the SAME machinery on the true reward reaches 0.91-0.97. Before concluding
# that the fitted reward is the whole story, run the recipe the published FB/HILP/TD-JEPA
# numbers were produced with, on our measure. If it also lands below the control, the
# wrong-reward floor is the story and the next campaign is features, not optimisation.
#
# Read off Factored-FB/impls/configs/critic/*.yaml and the TD-JEPA paper (Tab. 4/5, App. E):
# ensemble MEAN of two (NOT exact-min), flow-BC one-step DDPG actor with -Q/mean|Q| +
# alpha*BC, alpha 3 cube / 0.3 antmaze, tau 0.005, batch 256, 1M steps, 512x4 + LayerNorm.
#
# WHAT THE EXISTING CODE ALREADY GIVES, so nothing was added for this arm:
#   `actor_mode=ddpg` (the DEFAULT) IS that actor -- `flow_actor_loss` is literally
#   -Q/stopgrad(mean|q_ens|) + bc_coeff*distill + bc_flow_loss (agents/psmflow.py:1002-1005),
#   so bc_coeff IS alpha.
#   `pessimism_penalty=0` + `actor_pessimism_penalty=0` IS the ensemble mean.
#
# SECOND DEVIATION, FORCED, and it is theoretical rather than a config nit. The brief asks
# for the backup at the ACTOR's u (policy improvement). That is `policy_index=task_vector`,
# and `create` refuses it with psi_form=affine:
#
#   "the affine head reads its index slot as the POLICY latent u' and encodes it into w(u'),
#    while A and beta are by Assumption `affine` independent of the policy index. A z_dim
#    task vector in that slot would make A(s,u), beta(s,u) functions of the task."
#
# Confirmed in the sampler: under policy_index=latent the code builds the actor's latent for
# the bootstrap and then OVERWRITES it with the prior draw (:637-640), because the
# continuation at s' must be the same index the online side carries -- that is what makes
# G(s', u') an in-support decode (Prop. `insample`), and it is why backup_explore_frac is
# inert there. So affine + actor-u backup cannot both hold.
#
# Resolved by keeping AFFINE and the fixed-index backup, and dropping the actor-u backup.
# The arm exists to test FB's OPTIMISATION recipe on OUR measure; the backup index is part
# of the measure's definition, not of the optimisation recipe, so changing it would change
# what is under test. The alternative -- psi_form=free + task_vector -- is a different
# measure, and free psi measured 0.083 on cube.
#
# ONE DEVIATION FROM THE BRIEF, stated because it is not a free choice. The brief asks for
# BC toward `u_data` (the point preimage). `flow_actor_loss`'s BC term targets the actor's
# OWN CFM rollout, not u_data -- which is the flow-BC recipe FB actually uses, and u_data
# still enters one level down through `bc_flow_loss`, the CFM field's own regression onto
# it. Changing the target to u_data directly is new code and a different actor. Left as is.
#
# PRE-REGISTRATION (rewritten 2026-09-10 against the mechanism actually measured, not the
# one this arm was first drafted for; the drift-to-argmax path was refuted by the fixed-state
# probe and the mean-direction path by diag_task_vector_mean).
#
# What the diagnostics established. The fitted reward w^T phi is STATIONARY and WRONG, and
# the collapse is the value learning to optimise it BETTER over training. Arm C antmaze sd0:
# fitted return +3.02 -> +5.45 while success 0.360 -> 0.040; sd2: fitted flat 2.180 -> 2.185
# while success 0.660 -> 0.000. The selection rule is intact throughout (q_spread_rel flat or
# rising, top1-top2 gap flat, argmax not at the box edge), so this is not an argmax-noise
# problem. On cube, D1b collects 7.8x the fitted return of an agent that solves the task
# while succeeding 2% of the time.
#
# What fbparity is therefore testing: `alpha` (actor.bc_coeff) and the mean target are
# EXPLOITATION CONTROL. The BC pull is the only knob in this arm that limits how far the
# actor can chase the fitted reward away from the data.
#
#   antmaze fbparity PEAKS AND HOLDS instead of collapsing, and its fitted-return-vs-success
#     curve stops diverging  -> exploitation control is the fix, and the blurry reward is
#     survivable.
#   antmaze fbparity COLLAPSES the same way at alpha=0.3 -> do NOT conclude yet; alpha is the
#     only exploitation knob here, so re-run at alpha=1 and alpha=3 before reading a verdict.
#   below the control on both envs at every alpha -> the wrong-reward floor is the whole
#     story and the next campaign is features, not optimisation.
#
# Secondary, read off train.csv and eval500 as before: td_target_absmean flat over 1M;
# ladder step below 0.08; cube pooled above the control's 0.426; antmaze pooled above 0.23.
#
#   PSM_REPO=... PSM_DATA=... bash scripts/slurm/launch_fbparity.sh [table|smoke] [envkey]
set -euo pipefail
: "${PSM_REPO:?}"; : "${PSM_DATA:?}"
LOGDIR="$PSM_DATA/logs/slurm/fbparity"; mkdir -p "$LOGDIR"
SEEDS="${SEEDS:-0 1 2}"; STEPS="${STEPS:-1000000}"

# envkey | preimage npz | alpha (actor.bc_coeff), FB's per-domain value
ROWS=("cube|cube-single-play|3.0" "antmaze|antmaze-medium-navigate|0.3")

for row in "${ROWS[@]}"; do
  IFS='|' read -r envkey npz alpha <<<"$row"
  if [ -n "${2:-}" ] && [ "$2" != "$envkey" ]; then continue; fi
  NPZ="$PSM_DATA/preimages/${npz}.npz"
  GROUP="fbparity_${envkey}"
  OV="\
agent.pessimism_penalty=0.0 \
agent.actor_pessimism_penalty=0.0 \
agent.tau=0.005 \
agent.batch_size=256 \
agent.train_actor=true \
agent.acting=actor \
agent.actor_mode=ddpg \
agent.actor.bc_coeff=$alpha \
agent.actor.q_coeff=1.0 \
agent.actor.hidden_dim=512 \
agent.actor.hidden_layers=4 \
agent.actor.layer_norm=true \
agent.lr_actor=1.0e-4 \
agent.u_clip=1.5"
  if [ "${1:-}" = "table" ]; then
    echo "$GROUP  seeds='$SEEDS' steps=$STEPS  alpha(bc_coeff)=$alpha"
    echo "  npz $NPZ ; gamma from train_psmflow.sbatch's per-env default"
    echo "$OV" | tr ' ' '\n' | sed '/^$/d' | sed 's/^/  /'
    continue
  fi
  [ -f "$NPZ" ] || { echo "no such preimage npz: $NPZ" >&2; exit 1; }
  if [ "${1:-}" = "smoke" ]; then
    sbatch --partition=kisski-inference --account=general --gres=gpu:1 \
      --cpus-per-task=8 --mem=64G --time=00:40:00 --job-name="fbparity_smoke_${envkey}" \
      --output="$LOGDIR/%x-%j.out" --error="$LOGDIR/%x-%j.err" \
      --export=ALL,PSM_REPO="$PSM_REPO",PSM_DATA="$PSM_DATA",ENVKEY="$envkey",\
PREIMAGES="$NPZ",GROUP="fbparity_smoke_${envkey}",SEED=0,STEPS=200,\
EVAL_INT=200,EVAL_EPS=2,LOG_INT=50,SAVE_INT=1000000,EXTRA="$OV" \
      "$PSM_REPO/scripts/slurm/train_psmflow.sbatch"
    continue
  fi
  for s in $SEEDS; do
    sbatch --partition=kisski-inference --account=general --gres=gpu:1 \
      --cpus-per-task=8 --mem=64G --time=24:00:00 --job-name="${GROUP}_sd${s}" \
      --output="$LOGDIR/%x-%j.out" --error="$LOGDIR/%x-%j.err" \
      --export=ALL,PSM_REPO="$PSM_REPO",PSM_DATA="$PSM_DATA",ENVKEY="$envkey",\
PREIMAGES="$NPZ",GROUP="$GROUP",SEED="$s",STEPS="$STEPS",SAVE_INT=250000,EXTRA="$OV" \
      "$PSM_REPO/scripts/slurm/train_psmflow.sbatch"
  done
done
