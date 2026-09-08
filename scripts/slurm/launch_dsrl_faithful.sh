#!/usr/bin/env bash
# 2026-09-08 -- the FIRST structurally faithful DSRL-SAC arm on this substrate.
#
# Why this config and not `actor_mode=dsrl_sac` on the defaults: under the default
# policy_index=latent, `sample_step_inputs` computes the actor's bootstrap latent and then
# overwrites it with the prior draw u'. psi is then the successor measure of the
# constant-latent policy pi_{u'}, so an actor trained against it is doing policy
# improvement against a value function that is not Q^pi for any pi it converges to --
# SAC's actor step without SAC's policy evaluation step. policy_index=task_vector is the
# one index under which u_next survives into the backup, and psi_form=affine asserts
# policy_index=latent, so the bilinear head is given up here by necessity.
#
# Three further departures from what has been run before, each pinned to evidence:
#   bc_coeff=0        DSRL's premise is that the frozen decoder IS the constraint. (This
#                     is what the dsrl_* arms already used; stated so the arm is legible.)
#   actor.prior_init  the default init has per-dim latent std 1.88 at u_clip=3 against the
#                     flow's prior 1.0, so the policy starts OUTSIDE the support the flow
#                     was fitted on. prior_init starts it inside.
#   u_clip            DSRL recommends b_W in [0.5, 1.5] for OFFLINE and [1,3] for online.
#                     Every run here has used 3.0 -- the top of the ONLINE range. Two arms
#                     so the box change and the structural change are separable.
#
#   PSM_REPO=... PSM_DATA=... bash launch_dsrl_faithful.sh [smoke]
set -euo pipefail
: "${PSM_REPO:?}"; : "${PSM_DATA:?}"
ENVKEY=cube; FLOW=cube-single-play
LOGDIR="$PSM_DATA/logs/slurm/dsrlfaithful"; mkdir -p "$LOGDIR"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-500000}"

# EXTRA carries agent overrides ONLY: train_psmflow.sbatch already supplies agent=,
# flow_ckpt_*, preimage_path, env_name, offline_steps, save_dir, run_group and seed.
ARM="agent.actor_mode=dsrl_sac agent.policy_index=task_vector agent.psi_form=free \
agent.index_agg=max agent.actor.index_panel=0 agent.train_actor=true agent.acting=actor \
agent.actor.bc_coeff=0.0 agent.actor.prior_init=true agent.actor.prior_init_std=0.3"

for UC in 1.5 3.0; do
  TAG=$(echo "$UC" | tr -d .)
  GROUP="dsrlfaithful_cube_uc${TAG}"
  for s in $SEEDS; do
    NAME="${GROUP}_sd${s}"
    if [ "${1:-}" = "smoke" ]; then
      echo "SMOKE $NAME  u_clip=$UC steps=$STEPS"
      echo "   EXTRA=$ARM agent.u_clip=$UC"
      continue
    fi
    sbatch --partition=kisski-inference --account=general --gres=gpu:1 \
      --cpus-per-task=8 --mem=64G --time=08:00:00 --job-name="$NAME" \
      --output="$LOGDIR/%x-%j.out" --error="$LOGDIR/%x-%j.err" \
      --export=ALL,PSM_REPO="$PSM_REPO",PSM_DATA="$PSM_DATA",ENVKEY="$ENVKEY",\
PREIMAGES="$PSM_DATA/preimages/$FLOW.npz",GROUP="$GROUP",SEED="$s",STEPS="$STEPS",\
EXTRA="$ARM agent.u_clip=$UC" \
      "$PSM_REPO/scripts/slurm/train_psmflow.sbatch"
  done
done
