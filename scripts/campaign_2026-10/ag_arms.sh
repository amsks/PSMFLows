#!/usr/bin/env bash
# Advantage-guided actor arm (docs/design/2026-10-20-advantage-guided-actor.md): critic A's recipe
# (ja_arms.sh fbc3_pt) with the flowbc delta trained by regression on ag_k scored prior draws
# conditioned on their standardised point value, the TD bootstrap guided at ag_w_boot from a Polyak
# copy, acting guided at ag_w_eval.
#
#   SEEDS="0 1 2" STEPS=250000 bash scripts/campaign_2026-10/ag_arms.sh                      # psmgoal_ag_cube
#   SUFFIX=_wb025 AG_EXTRA="agent.ag_w_boot=0.25" SEEDS="0" bash scripts/campaign_2026-10/ag_arms.sh
#   SMOKE=1 bash scripts/campaign_2026-10/ag_arms.sh                                         # 200-step smoke job
REPO=${PSM_REPO:-/mnt/home/amohan/git/Austin/PSMFLows}; PSM_DATA=${PSM_DATA:-/mnt/home/amohan/psm-data}
SEEDS=${SEEDS:-"0 1 2"}
STEPS=${STEPS:-250000}
if [ "${SMOKE:-0}" = 1 ]; then RUN="STEPS=200,EVAL_INT=100,EVAL_EPS=2,LOG_INT=50,SAVE_INT=200"; PFX=smoke_; T=01:00:00
else RUN="STEPS=$STEPS,EVAL_INT=50000,EVAL_EPS=50,LOG_INT=5000,SAVE_INT=50000"; PFX=psmgoal_; T=${TIME:-10:00:00}; fi
COMMON="PSM_REPO=$REPO,PSM_DATA=$PSM_DATA,OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data,ENVKEY=cube,PREIMAGES=$PSM_DATA/preimages/cube-single-play.npz,$RUN"
JA_BASE="agent.policy_index=goal agent.bootstrap_source=actor agent.train_actor=true agent.actor_input=goal agent.goal_random_frac=0.0 agent.goal_cur_frac=0.0 agent.goal_sampling=geometric agent.measure_loss=softmax agent.acting=distill agent.eval_redistill=false agent.eval_goal_source=relabel"
# ag_k 16 / ag_pool 32 is the arm as run on aws and luis (03:55 CEST 10-20 relaunch; the 64 / 0 defaults cost 9x
# critic A's step). Set here so a control launched with only AG_EXTRA="agent.ag_w_boot=..." keeps them.
AG="$JA_BASE agent.actor_kind=flowbc agent.actor_value_kind=point agent.ag_actor=true agent.ag_k=16 agent.ag_pool=32 ${AG_EXTRA:-}"
for s in $SEEDS; do
  G=${PFX}ag${SUFFIX:-}_cube
  id=$(sbatch --parsable --partition=kisski-inference --account=general --time=$T --cpus-per-task=4 \
    --job-name=${G}_sd$s --output=$PSM_DATA/logs/slurm/%x-%j.out --error=$PSM_DATA/logs/slurm/%x-%j.err \
    --export="ALL,$COMMON,SEED=$s,GROUP=$G,EXTRA=$AG" $REPO/scripts/slurm/train_psmflow.sbatch)
  echo "$G sd$s job $id"
done
