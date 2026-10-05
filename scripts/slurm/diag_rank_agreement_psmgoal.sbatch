#!/usr/bin/env bash
#SBATCH --partition=kisski-inference
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# tools/diag_rank_agreement_psmgoal.py for one psmgoal seed against one critic arm, TASKS in turn.
# Needs SD (sd000|sd001|sd002), ARM (psmgoal_rhat|real), CSTEP (critic checkpoint step).
# Pairs psmgoal seed k with critic seed k. A task whose critic checkpoint is missing is skipped.
set -euo pipefail
export PSM_DATA=/mnt/home/amohan/psm-data
export PSM_REPO=/mnt/home/amohan/git/Austin/PSMFLows
export OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$PSM_REPO"
case "$ARM" in
  psmgoal_rhat) GROUP=psmgoal_rhat_critic ;;
  real) GROUP=dsrlna_real_critic ;;
  *) echo "unknown ARM $ARM" >&2; exit 1 ;;
esac
OUT=$PSM_DATA/logs/rank_agreement
RUN=$(ls -d $PSM_DATA/exp/PSMFLows/psmgoal_lift_cube_gc/${SD}_*)
for t in ${TASKS:-1 2 3 4 5}; do
  CDIR=$(ls -d $PSM_DATA/exp/PSMFLows/${GROUP}_task${t}/${SD}_* 2>/dev/null || true)
  if [ -z "$CDIR" ] || [ ! -f "$CDIR/params_${CSTEP}.pkl" ]; then
    echo "=== skip $ARM $SD task$t: no params_${CSTEP}.pkl"; continue
  fi
  NPZ=$PSM_DATA/logs/rollout_ranking/psmgoal_lift_cube_gc_${SD}_750k_task${t}.npz
  [ -f "$NPZ" ] || NPZ=null
  echo "=== $ARM $SD task$t critic=$CDIR rollout=$NPZ"
  .venv/bin/python tools/diag_rank_agreement_psmgoal.py agent=psmgoal \
    env_name=cube-single-play-singletask-task${t}-v0 \
    agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \
    agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz agent.use_point_preimage=true \
    agent.coef_source=regression restore_path="$RUN" restore_epoch=750000 \
    +critic_dir="$CDIR" +critic_epoch=$CSTEP +critic_arm=$ARM +rollout_npz=$NPZ +n_states=2000 \
    report_out=$OUT/cell_${ARM}_${CSTEP}_${SD}_task${t}.json \
    +scores_out=$OUT/scores_${ARM}_${CSTEP}_${SD}_task${t}.npz \
    hydra.run.dir=$OUT/hydra/${ARM}_${CSTEP}_${SD}_task${t}
done
