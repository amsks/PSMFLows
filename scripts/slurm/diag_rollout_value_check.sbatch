#!/usr/bin/env bash
# tools/diag_rollout_value_check.py on one GPU: five cube tasks for one (critic A, critic B) seed pair.
# sbatch --partition=kisski-inference --account=general \
#   --export=ALL,RUN_A=...,RUN_B=...,EPOCH=500000,TAG=...,EPS=8,TASKS="1 2 3 4 5",EXTRA="" \
#   scripts/slurm/diag_rollout_value_check.sbatch
#SBATCH --job-name=diag_rollout_value_check
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
set -euo pipefail
PSM_REPO="${PSM_REPO:-/mnt/home/amohan/git/Austin/PSMFLows}"
export PSM_DATA="${PSM_DATA:-/mnt/home/amohan/psm-data}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data
cd "$PSM_REPO"
OUTD=$PSM_DATA/logs/rollout_value_check; mkdir -p $OUTD
for t in ${TASKS:-1 2 3 4 5}; do
  srun "$PSM_REPO/.venv/bin/python" tools/diag_rollout_value_check.py agent=psmgoal \
    env_name=cube-single-play-singletask-task${t}-v0 \
    agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \
    agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz \
    agent.coef_source=hgoal_each agent.acting=gpi \
    restore_path=$RUN_A restore_epoch=$EPOCH +restore_path_b=$RUN_B \
    eval_episodes=${EPS:-8} seed=0 ${EXTRA:-} report_out=$OUTD/${TAG}_task${t}.json
done
