#!/usr/bin/env bash
# Critic A's recipe on antmaze with the TD latent drawn from each transition's stored posterior
# (agent.use_point_preimage=false) read from the preimage file regenerated with the two EM fixes
# ported from uoft-isl/inversion-noisy-q (init_at_preimage, cov_shrinkage 70; median ESS 130 against
# 5.6 in the original file). The point arm on the original file (psmgoal_ja_fbc3_pt_antmaze) is the
# control; the point preimage is the same in both files.
#
#   SEEDS="0 1 2" STEPS=500000 bash scripts/campaign_2026-10/antmaze_k70_arms.sh     # psmgoal_ja_fbc3_mix_k70_antmaze
#   SMOKE=1 bash scripts/campaign_2026-10/antmaze_k70_arms.sh                        # 200-step smoke job
REPO=${PSM_REPO:-/mnt/home/amohan/git/Austin/PSMFLows}; PSM_DATA=${PSM_DATA:-/mnt/home/amohan/psm-data}
SEEDS=${SEEDS:-"0 1 2"}
STEPS=${STEPS:-500000}
NPZ=$PSM_DATA/preimages/antmaze-medium-navigate-a20-ps1-ns10-N200-init-k70.npz
[ -f "$NPZ" ] || { echo "missing $NPZ (scripts/hf_preimages.py pull --name antmaze-medium-navigate-a20-ps1-ns10-N200-init-k70 --dest \$PSM_DATA)" >&2; exit 1; }
if [ "${SMOKE:-0}" = 1 ]; then RUN="STEPS=200,EVAL_INT=100,EVAL_EPS=2,LOG_INT=50,SAVE_INT=200"; PFX=smoke_; T=01:00:00
else RUN="STEPS=$STEPS,EVAL_INT=50000,EVAL_EPS=50,LOG_INT=5000,SAVE_INT=250000"; PFX=psmgoal_; T=${TIME:-16:00:00}; fi
COMMON="PSM_REPO=$REPO,PSM_DATA=$PSM_DATA,OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data,ENVKEY=antmaze,PREIMAGES=$NPZ,$RUN"
# critic A's recipe as run on antmaze (psmgoal_ja_fbc3_pt_antmaze: goal_discount 0.99, fb_hit_spec 0:2:0.5;
# the sbatch sets agent.discount=0.99 for ENVKEY=antmaze), with the posterior latent in place of the point.
A="agent.policy_index=goal agent.bootstrap_source=actor agent.train_actor=true agent.actor_input=goal agent.goal_random_frac=0.0 agent.goal_cur_frac=0.0 agent.goal_sampling=geometric agent.measure_loss=softmax agent.acting=distill agent.eval_redistill=false agent.eval_goal_source=relabel agent.actor_kind=flowbc agent.fb_bc_coeff=3.0 agent.actor_value_kind=point agent.goal_discount=0.99 agent.fb_hit_spec=0:2:0.5 agent.use_point_preimage=false ${K70_EXTRA:-}"
for s in $SEEDS; do
  G=${PFX}ja_fbc3_mix_k70${SUFFIX:-}_antmaze
  id=$(sbatch --parsable --partition=kisski-inference --account=general --time=$T --cpus-per-task=4 \
    --job-name=${G}_sd$s --output=$PSM_DATA/logs/slurm/%x-%j.out --error=$PSM_DATA/logs/slurm/%x-%j.err \
    --export="ALL,$COMMON,SEED=$s,GROUP=$G,EXTRA=$A" $REPO/scripts/slurm/train_psmflow.sbatch)
  echo "$G sd$s job $id"
done
