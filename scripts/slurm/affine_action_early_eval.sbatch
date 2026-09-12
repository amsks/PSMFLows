#!/usr/bin/env bash
# Re-evaluate saved 50k AntMaze checkpoints before the proposed freeze-phi study.
# sbatch [resource options] this-file SEED CAMPAIGN CODE
#SBATCH --job-name=psm-action-early-eval
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

SEED="${1:?training seed}"
CAMPAIGN="${2:?absolute campaign directory}"
CODE="${3:?immutable source snapshot}"
case "$SEED" in 0|1|2) ;; *) echo "seed must be 0, 1 or 2" >&2; exit 2 ;; esac
shopt -s nullglob
RUN_FLAGS=("$CAMPAIGN"/exp/*/"affine_action_train_antmaze_sd${SEED}"/*/flags.json)
test "${#RUN_FLAGS[@]}" -eq 1
RUN_DIR="${RUN_FLAGS[0]%/flags.json}"
PY="$CODE/.venv/bin/python"
test -f "$RUN_DIR/params_50000.pkl"
test -f "$CODE/source_manifest.json"
export PYTHONPATH="$CODE"
export PYTHONDONTWRITEBYTECODE=1
export OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export WANDB_MODE=offline
cd "$CODE"

"$PY" - "$CODE" "$RUN_DIR" "$SEED" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
root, run, seed = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
manifest = json.loads((root / 'source_manifest.json').read_text())
for name, expected in manifest['sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
flags = json.loads((run / 'flags.json').read_text())
assert flags['seed'] == seed
assert flags['agent']['measure_action_input'] == 'action'
assert flags['agent']['discount'] == .99
assert flags['agent']['acting'] == 'gpi'
assert not flags['agent']['train_actor']
assert not flags['agent']['dsrl_na']['enabled']
assert not flags['agent']['action_critic']['enabled']
print('Verified original checkpoint configuration and source manifest', flush=True)
PY

srun "$PY" tools/eval_checkpoint.py \
  agent=psmflow env_name=antmaze-medium-navigate-singletask-task1-v0 \
  restore_path="$RUN_DIR" restore_epoch=50000 seed=0 \
  eval_episodes=500 eval_workers=4 \
  report_out="$CAMPAIGN/reports/antmaze_sd${SEED}_task1_50000.json"
