#!/usr/bin/env bash
# Run from an immutable source snapshot. No existing checkpoints are overwritten.
# sbatch [resource flags] this-file smoke|train cube|antmaze SEED CAMPAIGN CODE DATA
# A successful train phase also runs all five tasks at the fixed 500k checkpoint.
#SBATCH --job-name=psm-affine-action
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

PHASE="${1:?smoke|train}"
ENVKEY="${2:?cube|antmaze}"
SEED="${3:?training seed}"
CAMPAIGN="${4:?absolute output directory}"
CODE="${5:?immutable source snapshot}"
DATA="${6:?artifact root}"
case "$PHASE" in
  smoke) STEPS=200; EVAL_INTERVAL=200; EVAL_EPISODES=2; LOG_INTERVAL=50; SAVE_INTERVAL=200 ;;
  train) STEPS=500000; EVAL_INTERVAL=50000; EVAL_EPISODES=50; LOG_INTERVAL=5000; SAVE_INTERVAL=50000 ;;
  *) echo "unknown phase: $PHASE" >&2; exit 2 ;;
esac
case "$ENVKEY" in
  cube) FLOW=cube-single-play; TRAIN_ENV=cube-single-play-singletask-v0; DISCOUNT=0.98 ;;
  antmaze) FLOW=antmaze-medium-navigate; TRAIN_ENV=antmaze-medium-navigate-singletask-v0; DISCOUNT=0.99 ;;
  *) echo "unknown environment: $ENVKEY" >&2; exit 2 ;;
esac
case "$SEED" in 0|1|2) ;; *) echo "seed must be 0, 1 or 2" >&2; exit 2 ;; esac

GROUP="affine_action_${PHASE}_${ENVKEY}_sd${SEED}"
PY="$CODE/.venv/bin/python"
FLOW_DIR="$DATA/flow/$FLOW"
PREIMAGES="$DATA/preimages/$FLOW.npz"
test -f "$FLOW_DIR/params_500000.pkl"
test -f "$PREIMAGES.meta.json"
test -f "$PREIMAGES"
test -f "$CODE/source_manifest.json"
mkdir -p "$CAMPAIGN/exp" "$CAMPAIGN/reports" "$CAMPAIGN/wandb"
export WANDB_MODE=offline
export WANDB_DIR="$CAMPAIGN/wandb"
export OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export PYTHONPATH="$CODE"
export PYTHONDONTWRITEBYTECODE=1
cd "$CODE"

# Refuse to run if the snapshot changed after submission.
"$PY" - "$CODE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / 'source_manifest.json').read_text())
for name, expected in manifest['sha256'].items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    assert actual == expected, f'source changed: {name}'
print('Verified immutable source manifest', flush=True)
PY

srun "$PY" main.py \
  agent=psmflow env_name="$TRAIN_ENV" seed="$SEED" run_group="$GROUP" \
  agent.measure_action_input=action agent.psi_form=affine agent.policy_index=latent \
  agent.train_actor=false agent.acting=gpi agent.discount="$DISCOUNT" \
  agent.flow_ckpt_path="$FLOW_DIR" agent.flow_ckpt_epoch=500000 \
  agent.preimage_path="$PREIMAGES" agent.use_point_preimage=true \
  agent.measure_u_samples=1 agent.gpi_num_u=64 agent.gpi_decode=onestep \
  offline_steps="$STEPS" online_steps=0 \
  eval_interval="$EVAL_INTERVAL" eval_episodes="$EVAL_EPISODES" \
  log_interval="$LOG_INTERVAL" save_interval="$SAVE_INTERVAL" save_dir="$CAMPAIGN/exp"

# Validate actual saved configuration and training completion before any evaluation.
"$PY" - "$CAMPAIGN" "$GROUP" "$SEED" "$STEPS" "$DISCOUNT" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path
root, group, seed, steps, discount = sys.argv[1:]
root = Path(root)
paths = list((root / 'exp').glob(f'*/{group}/*/flags.json'))
assert len(paths) == 1, f'expected one run, found {paths}'
p = paths[0]
f = json.loads(p.read_text())
assert f['seed'] == int(seed) and f['offline_steps'] == int(steps)
expected = dict(measure_action_input='action', psi_form='affine', policy_index='latent',
                train_actor=False, acting='gpi', measure_u_samples=1, gpi_num_u=64,
                discount=float(discount), gpi_decode='onestep', use_point_preimage=True)
for key, value in expected.items():
    assert f['agent'][key] == value, (key, f['agent'][key], value)
rows = list(csv.DictReader((p.parent / 'train.csv').open()))
assert int(rows[-1]['step']) == int(steps), rows[-1]['step']
for row in rows:
    for key in ('training/psm_loss', 'training/orth_loss'):
        assert math.isfinite(float(row[key])), (key, row[key])
assert (p.parent / f'params_{steps}.pkl').is_file()
out = dict(group=group, run_dir=str(p.parent), flags=f, completed_steps=int(steps),
           final_metrics=rows[-1])
(root / 'reports' / f'{group}_validation.json').write_text(json.dumps(out, indent=2))
print('Saved flags and completed training validated:', p.parent, flush=True)
PY

if [ "$PHASE" = smoke ]; then exit 0; fi

# Locate the unique validated run; keep argument boundaries intact.
shopt -s nullglob
RUN_FLAGS=("$CAMPAIGN"/exp/*/"$GROUP"/*/flags.json)
test "${#RUN_FLAGS[@]}" -eq 1
RUN_DIR="${RUN_FLAGS[0]%/flags.json}"
for TASK in 1 2 3 4 5; do
  srun "$PY" tools/eval_checkpoint.py \
    agent=psmflow env_name="$FLOW-singletask-task$TASK-v0" \
    restore_path="$RUN_DIR" restore_epoch=500000 seed=0 \
    eval_episodes=500 eval_workers=4 \
    report_out="$CAMPAIGN/reports/${ENVKEY}_sd${SEED}_task${TASK}_500000.json"
done
