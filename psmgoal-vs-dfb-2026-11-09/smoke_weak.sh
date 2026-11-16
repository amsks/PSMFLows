#!/usr/bin/env bash
# 200-step CPU smoke of each arm in arms_weak.tsv (same command as the 11-05 ladder smoke).
PSM_DATA=/mnt/home/amohan/psm-data; REPO=/mnt/home/amohan/git/Austin/PSMFLows; F=$REPO/psmgoal-vs-dfb-2026-11-09
OUT=$PSM_DATA/psmgoal-vs-dfb-2026-11-09/smoke_train; mkdir -p $OUT/logs
export JAX_PLATFORMS=cpu OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data MUJOCO_GL=egl WANDB_MODE=offline WANDB_DIR=$OUT/wandb PSM_DATA
cd $REPO
while IFS=$'\t' read -r arm extra; do
  [ -n "$arm" ] || continue
  ( .venv/bin/python main.py agent=psmgoal env_name=cube-single-play-singletask-v0 \
      agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \
      agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz $extra \
      offline_steps=200 eval_interval=100 eval_episodes=2 log_interval=50 save_interval=200 \
      save_dir=$OUT run_group=smoke_$arm seed=0 > $OUT/logs/$arm.log 2>&1; echo "rc=$?" > $OUT/logs/$arm.rc ) &
done < $F/arms_weak.tsv
wait
for f in $OUT/logs/*.rc; do echo "$(basename $f .rc) $(cat $f)"; done
