#!/bin/bash
# sbatch --export=ALL,KIND=psmgoal|dfb,RUN_DIR=<dir>,EPOCH=500000,TAG=<name> psmgoal-vs-dfb-2026-11-09/diag.sbatch
#SBATCH --job-name=critcmp
#SBATCH --output=/mnt/home/amohan/psm-data/psmgoal-vs-dfb-2026-11-09/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/psmgoal-vs-dfb-2026-11-09/slurm/%x-%j.err
#SBATCH --partition=kisski-inference
#SBATCH --account=general
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
cd /mnt/home/amohan/git/Austin/PSMFLows
export PSM_DATA=/mnt/home/amohan/psm-data OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV="${ENV:-cube-single-play-singletask-v0}"; FLOW="${FLOW:-cube-single-play}"
.venv/bin/python psmgoal-vs-dfb-2026-11-09/diag_critic_compare.py --kind "$KIND" --run_dir "$RUN_DIR" --env "$ENV" \
  --flow_ckpt "$PSM_DATA/flow/$FLOW" --preimage_path "$PSM_DATA/preimages/$FLOW.npz" \
  --epoch "$EPOCH" --report_out "$PSM_DATA/psmgoal-vs-dfb-2026-11-09/$TAG.json"
