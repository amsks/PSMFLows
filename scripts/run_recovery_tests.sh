#!/usr/bin/env bash
# Run the recovery diagnostics against the PUBLISHED flow + preimages, fetching them from
# Hugging Face first if they are not already under $PSM_DATA.
#
#   decode recovery    tools/validate_decode_recovery.py
#                      test 1: does a latent drawn from the stored preimage mixture decode
#                              back to the recorded action?
#                      test 2: do actions generated from prior latents exist anywhere in
#                              the buffer (full nearest-neighbour search)?
#   dynamics recovery  tools/validate_dynamics_recovery.py
#                      restore MuJoCo to the recorded state, step the decoded action, and
#                      compare where it lands against the recorded action replayed from
#                      the same state.
#
# Usage:
#   PSM_DATA=/path/for/big/files bash scripts/run_recovery_tests.sh [cube|antmaze|pointmaze|all]
#
#   SMOKE=1        tiny sizes -- checks the wiring in ~a minute, numbers not reportable
#   SKIP_FETCH=1   the artifacts are already in place
#   TESTS="decode dynamics"   which of the two to run
#   GPU=0          CUDA_VISIBLE_DEVICES (unset leaves the default; the tools run on CPU too)
#
# Reports land in $PSM_DATA/logs/<tool>_<name>.json, one JSON per tool per environment.
set -euo pipefail

WHICH="${1:-all}"
: "${PSM_DATA:?PSM_DATA=<dir holding preimages/ and flow/> required}"
read -r -a TESTS <<< "${TESTS:-decode dynamics}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || PY=python
LOGS="$PSM_DATA/logs"
mkdir -p "$LOGS"

case "$WHICH" in
  cube)      KEYS=(cube) ;;
  antmaze)   KEYS=(antmaze) ;;
  pointmaze) KEYS=(pointmaze) ;;
  all)       KEYS=(cube antmaze pointmaze) ;;
  *) echo "unknown target: $WHICH (cube|antmaze|pointmaze|all)" >&2; exit 1 ;;
esac

if [ -z "${SKIP_FETCH:-}" ]; then
  bash "$REPO/scripts/fetch_preimages_hf.sh" "$WHICH"
fi

cd "$REPO"
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$PSM_DATA/ogbench}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"          # headless: pointmaze builds a renderer
export XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU"

# Sizes. The decode test scores `preimage_limit` rows; the buffer search scans all ~1M
# rows regardless of how many states query it; the dynamics test costs ~15 ms per state.
if [ -n "${SMOKE:-}" ]; then
  ROWS=512;   T2_STATES=32;  T2_LATENTS=4;  DYN_STATES=16
else
  ROWS=50000; T2_STATES=512; T2_LATENTS=32; DYN_STATES=512
fi

for key in "${KEYS[@]}"; do
  case "$key" in
    cube)      ENV_NAME=cube-single-play-singletask-v0;                 NAME=cube-single-play ;;
    antmaze)   ENV_NAME=antmaze-medium-navigate-singletask-v0;          NAME=antmaze-medium-navigate ;;
    pointmaze) ENV_NAME=pointmaze-medium-navigate-singletask-task1-v0;  NAME=pointmaze-medium-navigate ;;
  esac
  NPZ="$PSM_DATA/preimages/$NAME.npz"
  FLOW="$PSM_DATA/flow/$NAME"
  [ -f "$NPZ" ] || { echo "no such npz: $NPZ (run scripts/fetch_preimages_hf.sh)" >&2; exit 1; }

  # The published flow is the 500k checkpoint inverted at flow_steps=100; both tools
  # assert this against the sidecar rather than trusting the numbers below.
  EPOCH="${FLOW_EPOCH:-500000}"
  STEPS="${FLOW_STEPS:-100}"
  COMMON=(agent=fql env_name="$ENV_NAME" restore_path="$FLOW" restore_epoch="$EPOCH"
          agent.flow_steps="$STEPS" +preimage_npz="$NPZ")

  for test in "${TESTS[@]}"; do
    case "$test" in
      decode)
        echo "== decode recovery: $NAME"
        "$PY" tools/validate_decode_recovery.py "${COMMON[@]}" \
          preimage_limit="$ROWS" +t2_states="$T2_STATES" +t2_latents="$T2_LATENTS" \
          "+t1_sources=[mixture,point,prior]" \
          report_out="$LOGS/decode_recovery_$NAME.json" \
          hydra.run.dir="$LOGS/hydra/decode_$NAME" ;;
      dynamics)
        echo "== dynamics recovery: $NAME"
        "$PY" tools/validate_dynamics_recovery.py "${COMMON[@]}" \
          +n_states="$DYN_STATES" "+latent_sources=[mixture,point,prior]" \
          report_out="$LOGS/dynamics_recovery_$NAME.json" \
          hydra.run.dir="$LOGS/hydra/dynamics_$NAME" ;;
      *) echo "unknown test: $test (decode|dynamics)" >&2; exit 1 ;;
    esac
  done
done

echo
echo "reports:"
ls -1 "$LOGS"/*_recovery_*.json
