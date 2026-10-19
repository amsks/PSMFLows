#!/usr/bin/env bash
# Reward-index arm (docs/design/2026-10-19-reward-index.md): Density FB's z arm in critic A's recipe.
# psmgoal_ri_fbc3_cube x seeds 0-2: policy_index=reward (w = centred ridge of the hindsight kernel reward on the
# projected-phi feature f, half the rows random), critic A's softmax loss, the flowbc actor at w (fb_bc_coeff 3),
# bootstrap from that actor at (s', w). Test time coef_source=ridge; readouts distill (actor at w), gpi
# (best-of-64 by the goal mass at w) and gpi + ridge_score=reward (best-of-64 by the expected relabel reward).
#
#   SEEDS="0 1 2" STEPS=250000 bash scripts/campaign_2026-10/reward_index_arms.sh    # submit; prints job ids
#   SMOKE=1 bash scripts/campaign_2026-10/reward_index_arms.sh                       # 200-step smokes instead
REPO=${PSM_REPO:-/mnt/home/amohan/git/Austin/PSMFLows}; PSM_DATA=${PSM_DATA:-/mnt/home/amohan/psm-data}
SEEDS=${SEEDS:-"0 1 2"}
STEPS=${STEPS:-250000}
if [ "${SMOKE:-0}" = 1 ]; then RUN="STEPS=200,EVAL_INT=100,EVAL_EPS=2,LOG_INT=50,SAVE_INT=200"; PFX=smoke_; T=01:00:00
else RUN="STEPS=$STEPS,EVAL_INT=50000,EVAL_EPS=50,LOG_INT=5000,SAVE_INT=50000"; PFX=psmgoal_; T=${TIME:-10:00:00}; fi
COMMON="PSM_REPO=$REPO,PSM_DATA=$PSM_DATA,OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data,ENVKEY=cube,PREIMAGES=$PSM_DATA/preimages/cube-single-play.npz,$RUN"
# critic A's recipe (ja_arms.sh fbc3_pt) with the goal index replaced by the reward index; the actor takes w.
RI="agent.policy_index=reward agent.bootstrap_source=actor agent.train_actor=true agent.actor_input=w agent.goal_random_frac=0.0 agent.goal_cur_frac=0.0 agent.goal_sampling=geometric agent.measure_loss=softmax agent.acting=distill agent.eval_redistill=false agent.actor_kind=flowbc agent.fb_bc_coeff=3.0 agent.coef_source=ridge"
for s in $SEEDS; do
  G=${PFX}ri_fbc3_cube
  id=$(sbatch --parsable --partition=kisski-inference --account=general --time=$T --cpus-per-task=4 \
    --job-name=${G}_sd$s --output=$PSM_DATA/logs/slurm/%x-%j.out --error=$PSM_DATA/logs/slurm/%x-%j.err \
    --export="ALL,$COMMON,SEED=$s,GROUP=$G,EXTRA=$RI" $REPO/scripts/slurm/train_psmflow.sbatch)
  echo "$G sd$s job $id"
done
