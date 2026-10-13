#!/usr/bin/env bash
# tools/diag_measure_pair_ranking.py on one GPU: a 64-state smoke, then the full run with ARGS.
# sbatch --partition=kisski-inference --account=general --export=ALL,ARGS="..." scripts/slurm/diag_measure_pair_ranking.sbatch
#SBATCH --job-name=diag_measure_pair_ranking
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -euo pipefail
PSM_REPO="${PSM_REPO:-/mnt/home/amohan/git/Austin/PSMFLows}"
export PSM_DATA="${PSM_DATA:-/mnt/home/amohan/psm-data}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl
cd "$PSM_REPO"
echo "node=$(hostname) args=${ARGS:-}"
srun "$PSM_REPO/.venv/bin/python" tools/diag_measure_pair_ranking.py --seeds 0 --steps 250000 --ks 20 --n-states 64 \
  --report-out "$PSM_DATA/logs/smoke_diag_measure_pair_ranking.json"
srun "$PSM_REPO/.venv/bin/python" tools/diag_measure_pair_ranking.py ${ARGS:-}
