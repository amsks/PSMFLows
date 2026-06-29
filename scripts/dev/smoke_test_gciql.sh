#!/usr/bin/env bash
# Smoke: tiny GCIQL run for state + visual. Verifies the vendored OGBench +
# isolated JAX env + data path work end-to-end. Uses tmpfs (/dev/shm) by default; override storage=nvme for the NVMe.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

JAX_PY="${JAX_PY:-/dev/shm/.venv-jax/bin/python}"
echo "[smoke-gciql] JAX device check:"
"$JAX_PY" -c "import jax; print('jax devices:', jax.devices())"

for VARIANT in state visual; do
    echo "[smoke-gciql] === $VARIANT ==="
    python run_gciql.py --config-name "cube_single_${VARIANT}" \
        train_steps=2000 log_interval=1000 eval_interval=1000 \
        save_interval=2000 eval_episodes=2 wandb_mode=offline

    RUN_DIR=$(ls -1dt /dev/shm/gciql_outputs/OGBench/*/* 2>/dev/null | head -1)
    echo "[smoke-gciql] inspecting $RUN_DIR"
    test -f "$RUN_DIR/eval.csv" || { echo "FAIL: no eval.csv"; exit 1; }
    grep -q "overall_success\|success" "$RUN_DIR/eval.csv" \
        || { echo "FAIL: no success metric logged"; exit 1; }
    ls "$RUN_DIR"/params_*.pkl >/dev/null 2>&1 \
        || { echo "FAIL: no checkpoint written"; exit 1; }
    echo "[smoke-gciql] $VARIANT PASS"
done
echo "[smoke-gciql] ALL PASS"
