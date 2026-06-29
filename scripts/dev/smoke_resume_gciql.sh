#!/usr/bin/env bash
# scripts/dev/smoke_resume_gciql.sh — verify GCIQL restore continues a run.
# Phase 1 runs a tiny GCIQL job that writes params_<epoch>.pkl; phase 2 restarts
# with restore_path/restore_epoch (+ exp_name to pin the same dir) and asserts a
# checkpoint past the restore epoch appears (GCIQL_STEP_OFFSET applied). The core
# resume machinery lives in run_gciql.py + vendored OGBench main.py:86-87.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
# run_gciql.py is the outer orchestrator (needs hydra) -> repo .venv python; it
# spawns the JAX env (JAX_PY) for the vendored main.py internally. A tmux/cron
# non-login shell has no `python` on PATH, so use the absolute interpreter.
PY="${PYTHON:-$REPO/.venv/bin/python}"
JAX_PY="${JAX_PY:-/dev/shm/.venv-jax/bin/python}"   # memory: jax venv is volatile /dev/shm
CONFIG="${CONFIG:-cube_single_state}"
EXP="smoke_resume_gciql_$$"

echo "[smoke-gciql-resume] JAX devices:"
"$JAX_PY" -c "import jax; print(jax.devices())"

echo "[smoke-gciql-resume] phase 1: fresh run (2000 steps, ckpt every 1000)"
"$PY" run_gciql.py --config-name "$CONFIG" \
    train_steps=2000 log_interval=1000 eval_interval=1000 \
    save_interval=1000 eval_episodes=2 wandb_mode=offline ++exp_name="$EXP"

# run_gciql saves to <output_root>/<entity>/<run_group>/<exp_name>/params_<N>.pkl;
# exp_name is unique ($$), so locate the dir holding this run's checkpoints.
RUN_DIR=$(dirname "$(ls -1t /dev/shm/gciql_outputs/*/*/"$EXP"/params_*.pkl 2>/dev/null | head -1)")
[ -d "$RUN_DIR" ] || { echo "FAIL: could not locate run dir for $EXP"; exit 1; }
echo "[smoke-gciql-resume] RUN_DIR=$RUN_DIR"
ls "$RUN_DIR"/params_*.pkl || { echo "FAIL: no params written"; exit 1; }
_epochs() { ls "$RUN_DIR"/params_*.pkl | sed -E 's/.*params_([0-9]+)\.pkl/\1/' | sort -n; }
RESTORE_EPOCH=$(_epochs | tail -1)
echo "[smoke-gciql-resume] restore from epoch $RESTORE_EPOCH"

echo "[smoke-gciql-resume] phase 2: resume +1000 steps (offset applied)"
"$PY" run_gciql.py --config-name "$CONFIG" \
    train_steps=1000 log_interval=1000 eval_interval=1000 \
    save_interval=1000 eval_episodes=2 wandb_mode=offline ++exp_name="$EXP" \
    ++restore_path="$RUN_DIR" ++restore_epoch="$RESTORE_EPOCH"

MAXEP=$(_epochs | tail -1)
echo "[smoke-gciql-resume] max epoch after resume: $MAXEP (restore was $RESTORE_EPOCH)"
[ "$MAXEP" -gt "$RESTORE_EPOCH" ] || { echo "FAIL: no checkpoint past restore epoch"; exit 1; }
echo "[smoke-gciql-resume] PASS ($CONFIG)  RUN_DIR=$RUN_DIR"
