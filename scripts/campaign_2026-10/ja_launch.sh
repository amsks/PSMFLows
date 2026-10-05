#!/usr/bin/env bash
# Launch the four joint-actor arms (2026-10-05), 3 seeds each, cube, 500k steps.
# PACK=3 (default): one packed job per arm (scripts/slurm/train_psmflow_packed.sbatch, 3 seeds on
# one GPU, 24G). PACK=1: one job per seed (train_psmflow.sbatch, MEM may be lowered to land on
# memory-blocked nodes). Prints every job id; records them in $PSM_DATA/logs/ja_launch_jobs.txt.
set -euo pipefail
PSM_DATA=/mnt/home/amohan/psm-data; REPO=/mnt/home/amohan/git/Austin/PSMFLows
source $PSM_DATA/gc_scripts/ja_arms.sh
PACK="${PACK:-3}"; MEM="${MEM:-}"; TIME="${TIME:-24:00:00}"
PRE=$PSM_DATA/preimages/cube-single-play.npz
COMMON="PSM_REPO=$REPO,PSM_DATA=$PSM_DATA,OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data,ENVKEY=cube,PREIMAGES=$PRE,STEPS=500000,EVAL_INT=50000,EVAL_EPS=50,LOG_INT=5000,SAVE_INT=250000"
for arm in psmgoal_ja_fbc3_sh_cube psmgoal_ja_fbc0_sh_cube psmgoal_ja_dsrl_sh_cube psmgoal_ja_fbc3_pt_cube; do
  if [ "$PACK" = 3 ]; then
    id=$(sbatch --parsable --partition=kisski-inference --account=general --time=$TIME ${MEM:+--mem=$MEM} \
      --job-name=${arm}_p3 --output=$PSM_DATA/logs/slurm/%x-%j.out --error=$PSM_DATA/logs/slurm/%x-%j.err \
      --export="ALL,$COMMON,SEEDS=0 1 2,GROUP=$arm,EXTRA=${JA_ARMS[$arm]}" $REPO/scripts/slurm/train_psmflow_packed.sbatch)
    echo "$arm seeds 0 1 2 (packed) job $id" | tee -a $PSM_DATA/logs/ja_launch_jobs.txt
  else
    for s in 0 1 2; do
      id=$(sbatch --parsable --partition=kisski-inference --account=general --time=$TIME ${MEM:+--mem=$MEM} --cpus-per-task=4 \
        --job-name=${arm}_sd$s --output=$PSM_DATA/logs/slurm/%x-%j.out --error=$PSM_DATA/logs/slurm/%x-%j.err \
        --export="ALL,$COMMON,SEED=$s,GROUP=$arm,EXTRA=${JA_ARMS[$arm]}" $REPO/scripts/slurm/train_psmflow.sbatch)
      echo "$arm seed $s job $id" | tee -a $PSM_DATA/logs/ja_launch_jobs.txt
    done
  fi
done
