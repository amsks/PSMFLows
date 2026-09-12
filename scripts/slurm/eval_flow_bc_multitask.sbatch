#!/usr/bin/env bash
# Re-evaluate all five AntMaze tasks with the same frozen one-step flow.
# sbatch [resource flags] this-file CODE FLOW_DIR REPORT_DIR
#SBATCH --job-name=psm-bc-antmaze-five-tasks
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
CODE="${1:?immutable source snapshot}"
FLOW_DIR="${2:?Stage-A checkpoint directory}"
REPORT_DIR="${3:?new report directory}"
test ! -e "$REPORT_DIR"
mkdir -p "$REPORT_DIR"
PY="$CODE/.venv/bin/python"
test -f "$FLOW_DIR/params_500000.pkl"
export PYTHONPATH="$CODE"
export PYTHONDONTWRITEBYTECODE=1
export OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export WANDB_MODE=offline
cd "$CODE"
"$PY" - "$FLOW_DIR" <<'PY'
import json
import sys
from pathlib import Path
flags = json.loads((Path(sys.argv[1]) / 'flags.json').read_text())
assert flags['agent']['agent_name'] == 'fql'
assert flags['agent']['bc_only']
print('Verified frozen BC checkpoint configuration', flush=True)
PY
for TASK in 1 2 3 4 5; do
  srun "$PY" tools/eval_checkpoint.py \
    agent=fql env_name="antmaze-medium-navigate-singletask-task${TASK}-v0" \
    restore_path="$FLOW_DIR" restore_epoch=500000 seed=0 \
    eval_episodes=500 eval_workers=4 \
    report_out="$REPORT_DIR/task${TASK}.json"
done
