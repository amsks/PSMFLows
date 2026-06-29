#!/usr/bin/env bash
# Smoke test: tiny cube_single run with save_eval_videos=true.
# Verifies per-run artifact layout (checkpoints + eval_videos) end-to-end.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DATA_PATH="${DATA_PATH:-/dev/shm/datasets}"

echo "[smoke] launching short cube_single run..."
python train.py \
    domain=cube_single \
    seed=0 \
    data_path="$DATA_PATH" \
    num_train_steps=100 \
    eval_every=50 \
    save_every=100 \
    log_every=100 \
    eval_n_episodes=2 \
    eval_relabel_size=64 \
    load_n_episodes=4 \
    save_eval_videos=true \
    use_wandb=false

# Find the most-recent outputs/ dir from this run.
LATEST_DIR=$(ls -1dt outputs/*cube-single*s0* 2>/dev/null | head -1)
echo "[smoke] inspecting $LATEST_DIR"

echo ""
echo "[smoke] checkpoints:"
ls -la "$LATEST_DIR/checkpoints/" || { echo "FAIL: no checkpoints/"; exit 1; }

echo ""
echo "[smoke] eval_videos:"
find "$LATEST_DIR/eval_videos/" -type f -name '*.gif' | sort

EXPECTED_GIFS=15   # 3 eval rounds (step 0, 50, 100) * 5 tasks
ACTUAL_GIFS=$(find "$LATEST_DIR/eval_videos/" -type f -name '*.gif' | wc -l)
if [ "$ACTUAL_GIFS" -ne "$EXPECTED_GIFS" ]; then
    echo "FAIL: expected $EXPECTED_GIFS GIFs, got $ACTUAL_GIFS"
    exit 1
fi

# Spot-check one GIF is a valid image.
SAMPLE_GIF=$(find "$LATEST_DIR/eval_videos/" -type f -name '*.gif' | head -1)
python -c "
import imageio.v3 as iio
arr = iio.imread('$SAMPLE_GIF')
assert arr.ndim == 4 and arr.shape[-1] == 3, f'bad shape: {arr.shape}'
assert arr.dtype.name == 'uint8', f'bad dtype: {arr.dtype}'
print(f'OK $SAMPLE_GIF shape={arr.shape}')
"

echo ""
echo "[smoke] PASS — $LATEST_DIR contains checkpoints/ and $ACTUAL_GIFS GIFs"
