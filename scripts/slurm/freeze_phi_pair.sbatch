#!/usr/bin/env bash
# One matched pair per GPU, sharing each training batch in one process.
# sbatch [resource flags] this-file smoke|pilot CAMPAIGN CODE SOURCE_RUN
#SBATCH --job-name=psm-freeze-phi-pair
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

PHASE="${1:?smoke|pilot}"
CAMPAIGN="${2:?absolute campaign output directory}"
CODE="${3:?immutable source snapshot}"
SOURCE_RUN="${4:?original seed0 run containing params_50000.pkl}"
case "$PHASE" in
  smoke) ADDITIONAL=200; INTERVAL=200; EPISODES=2; LOG_INTERVAL=50; ENDPOINT=50200 ;;
  pilot) ADDITIONAL=450000; INTERVAL=50000; EPISODES=50; LOG_INTERVAL=5000; ENDPOINT=500000 ;;
  *) echo "unknown phase: $PHASE" >&2; exit 2 ;;
esac
PY="$CODE/.venv/bin/python"
OUT="$CAMPAIGN/$PHASE"
test ! -e "$OUT"
test -f "$SOURCE_RUN/params_50000.pkl"
test -f "$CODE/source_manifest.json"
mkdir -p "$CAMPAIGN/reports"
export PYTHONPATH="$CODE"
export PYTHONDONTWRITEBYTECODE=1
export OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export WANDB_MODE=offline
cd "$CODE"

"$PY" - "$CODE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / 'source_manifest.json').read_text())
for name, expected in manifest['sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
print('Verified source manifest', flush=True)
PY

srun "$PY" tools/train_freeze_phi_pair.py \
  --source-run "$SOURCE_RUN" --source-step 50000 --output-dir "$OUT" \
  --additional-steps "$ADDITIONAL" --save-interval "$INTERVAL" \
  --eval-interval "$INTERVAL" --eval-episodes "$EPISODES" \
  --log-interval "$LOG_INTERVAL" --eval-relabel-size 10000 --seed 0

"$PY" - "$OUT" "$ENDPOINT" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path
root, endpoint = Path(sys.argv[1]), int(sys.argv[2])
assert (root / 'completion.json').is_file()
for arm, expected_train_phi in (('control', True), ('frozen', False)):
    directory = root / arm
    flags = json.loads((directory / 'flags.json').read_text())
    assert flags['agent']['train_phi'] == expected_train_phi
    assert flags['agent']['measure_action_input'] == 'action'
    assert flags['agent']['discount'] == .99
    assert flags['agent']['acting'] == 'gpi'
    assert not flags['agent']['train_actor']
    with (directory / 'train.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert int(rows[-1]['step']) == endpoint
    for row in rows:
        for key, value in row.items():
            if key.startswith('training/') and value:
                assert math.isfinite(float(value)), (arm, key, value)
    assert (directory / f'params_{endpoint}.pkl').is_file()
print('Both arm endpoints validated', flush=True)
PY

if [ "$PHASE" = smoke ]; then exit 0; fi
for ARM in control frozen; do
  for TASK in 1 2 3 4 5; do
    srun "$PY" tools/eval_checkpoint.py \
      agent=psmflow env_name="antmaze-medium-navigate-singletask-task${TASK}-v0" \
      restore_path="$OUT/$ARM" restore_epoch=500000 seed=0 \
      eval_episodes=500 eval_workers=4 \
      report_out="$CAMPAIGN/reports/${ARM}_task${TASK}_500000.json"
  done
done
