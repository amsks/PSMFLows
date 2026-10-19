#!/usr/bin/env bash
# Pool-measure arms (docs/design/2026-10-19-pool-measure.md, docs/plans/2026-10-19-pool-measure.md Task 5).
# Three cube arms x seeds 0-2, 500k steps, 24 h: the measure trained as a distribution over a fixed 1024-state
# reference set (measure_loss=pool), with (bank) a one-member BC bank as the policy index, (fdplus) arm A's goal
# index with the frozen +step field from A sd000 500k, (ja) arm A's goal index with the live flowbc actor.
# Smokes 2525741-2525743 (200 steps) exit 0; flags pool / 1024 / hash dfea8933...; eval path checked 2525745-48.
#
#   SEEDS="0 1 2" bash scripts/campaign_2026-10/pool_arms.sh            # submit; prints job ids
#   SMOKE=1 bash scripts/campaign_2026-10/pool_arms.sh                  # 200-step smokes instead
REPO=${PSM_REPO:-/mnt/home/amohan/git/Austin/PSMFLows}; PSM_DATA=${PSM_DATA:-/mnt/home/amohan/psm-data}
A=$PSM_DATA/exp/PSMFLows/psmgoal_ja_fbc3_pt_cube/sd000_s_2522773.0.20261006_111542
SEEDS=${SEEDS:-"0 1 2"}
if [ "${SMOKE:-0}" = 1 ]; then RUN="STEPS=200,EVAL_INT=100,EVAL_EPS=2,LOG_INT=50,SAVE_INT=200"; PFX=smoke_; T=01:00:00
else RUN="STEPS=500000,EVAL_INT=50000,EVAL_EPS=50,LOG_INT=5000,SAVE_INT=250000"; PFX=psmgoal_; T=24:00:00; fi
COMMON="PSM_REPO=$REPO,PSM_DATA=$PSM_DATA,OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data,ENVKEY=cube,PREIMAGES=$PSM_DATA/preimages/cube-single-play.npz,$RUN"
# arm A's recipe (scripts/campaign_2026-10/ja_arms.sh fbc3_pt) without its measure_loss, which the pool replaces
JA_PT="agent.policy_index=goal agent.train_actor=true agent.actor_input=goal agent.goal_random_frac=0.0 agent.goal_cur_frac=0.0 agent.goal_sampling=geometric agent.acting=distill agent.eval_redistill=false agent.eval_goal_source=relabel agent.actor_kind=flowbc agent.fb_bc_coeff=3.0 agent.actor_value_kind=point"
declare -A EXTRA=(
  [bank]="agent.measure_loss=pool agent.policy_index=bank agent.train_actor=false agent.acting=gpi agent.coef_source=pool_reward"
  [fdplus]="$JA_PT agent.measure_loss=pool agent.bootstrap_source=frozen_delta agent.frozen_delta_path=$A agent.frozen_delta_epoch=500000 agent.frozen_delta_mode=plus"
  [ja]="$JA_PT agent.measure_loss=pool agent.bootstrap_source=actor"
)
for arm in bank fdplus ja; do for s in $SEEDS; do
  G=${PFX}pool_${arm}_cube
  id=$(sbatch --parsable --partition=kisski-inference --account=general --time=$T --cpus-per-task=4 \
    --job-name=${G}_sd$s --output=$PSM_DATA/logs/slurm/%x-%j.out --error=$PSM_DATA/logs/slurm/%x-%j.err \
    --export="ALL,$COMMON,SEED=$s,GROUP=$G,EXTRA=${EXTRA[$arm]}" $REPO/scripts/slurm/train_psmflow.sbatch)
  echo "$G sd$s job $id"
done; done
