#!/usr/bin/env bash
# tools/diag_psmgoal_rollout_ranking.py on one GPU: records the 64 candidate actions, their scores
# and the executed pick at every eval step, for the five cube tasks of one checkpoint.
# sbatch --partition=kisski-inference --account=general --export=ALL,RUN_DIR=...,EPOCH=500000,TAG=...,EPS=20,EXTRA="..." \
#   scripts/slurm/diag_rollout_candidates.sbatch
#SBATCH --job-name=diag_rollout_candidates
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -euo pipefail
PSM_REPO="${PSM_REPO:-/mnt/home/amohan/git/Austin/PSMFLows}"
export PSM_DATA="${PSM_DATA:-/mnt/home/amohan/psm-data}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data
cd "$PSM_REPO"
OUTD=$PSM_DATA/logs/rollout_candidates; mkdir -p $OUTD
for t in ${TASKS:-1 2 3 4 5}; do
  srun "$PSM_REPO/.venv/bin/python" tools/diag_psmgoal_rollout_ranking.py agent=psmgoal \
    env_name=cube-single-play-singletask-task${t}-v0 \
    agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \
    agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz \
    restore_path=$RUN_DIR restore_epoch=$EPOCH eval_episodes=${EPS:-20} seed=0 ${EXTRA:-} \
    report_out=$OUTD/${TAG}_task${t}.json +npz_out=$OUTD/${TAG}_task${t}.npz
done
