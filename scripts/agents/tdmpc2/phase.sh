#!/usr/bin/env bash
# scripts/agents/tdmpc2/phase.sh — phase-probe over all TD-MPC2 (state) seeds.
#
# TD-MPC2 is a PyTorch model-based (MPC) agent exposing the same .act(obs, z)
# interface as FB/RLDP, so it reuses the generic torch phase-probe driver
# (scripts/probes/phase_probe.py). Two TD-MPC2-specific paths are handled inside that
# driver and evals/phase_probe.py, both gated on capability (FB-safe):
#   * goal context comes from eval_context(env=...) (cube goal xyz), not the
#     offline buffer's _infer_z;
#   * agent.reset() is called per episode to clear the MPPI warm-start so each
#     rollout plans fresh.
# For each __sN run under RESULTS_ROOT it picks final.pt and that run's saved
# .hydra/config.yaml. The phase classification itself is policy-agnostic
# (mechanical thresholds on effector/cube/gripper signals), identical to the
# other methods, so the MBRL row is directly comparable.
USAGE="usage: bash scripts/agents/tdmpc2/phase.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4 5 6 7 8 9'   seeds to process (default all 10)
    EPISODES=10                    rollouts per (task, scenario)
    SCENARIOS=S0,S1,S2             counterfactual scenarios (M1+M2)
    OUT_ROOT=analysis/probes/phase_probe_tdmpc2
    RESULTS_ROOT=/dev/shm/factored-fb/runs   run root (10-seed group lives here)
    RUN_GLOB='*tdmpc2_state_10seeds*'         per-seed dir glob (before __sN)
    DATA_PATH=/dev/shm/factored-fb/datasets   offline buffer root
    MUJOCO_GL=egl                  GL backend (egl on Linux)
    PY=.venv/bin/python
    DRY_RUN=1                      preview commands only"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
EPISODES="${EPISODES:-10}"
SCENARIOS="${SCENARIOS:-S0,S1,S2}"
OUT_ROOT="${OUT_ROOT:-analysis/probes/phase_probe_tdmpc2}"
RESULTS_ROOT="${RESULTS_ROOT:-/dev/shm/factored-fb/runs}"
RUN_GLOB="${RUN_GLOB:-*tdmpc2_state_10seeds*}"
DATA_PATH="${DATA_PATH:-/dev/shm/factored-fb/datasets}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
PY="${PY:-$REPO/.venv/bin/python}"

mkdir -p "$OUT_ROOT"
for s in $SEEDS; do
    sdir=$(ls -d "${RESULTS_ROOT}"/${RUN_GLOB}__s${s} 2>/dev/null | head -1)
    if [[ -z "$sdir" ]]; then
        echo "[phase] seed=$s: NO RESULTS DIR — skipping"; continue
    fi
    ckpt="${sdir}/checkpoints/final.pt"
    cfg="${sdir}/.hydra/config.yaml"
    if [[ ! -f "$ckpt" || ! -f "$cfg" ]]; then
        echo "[phase] seed=$s: missing ckpt/config in $sdir — skipping"; continue
    fi
    out="${OUT_ROOT}/s${s}_final"
    cmd=("$PY" scripts/probes/phase_probe.py
         --config "$cfg" --checkpoint "$ckpt"
         --seed "$s" --out "$out" --n-episodes "$EPISODES"
         --scenarios "$SCENARIOS" --data-path "$DATA_PATH"
         --device cuda --mujoco-gl "$MUJOCO_GL")
    echo "[phase] seed=$s out=$out"
    if _dry_run_preview "${cmd[@]}"; then continue; fi
    "${cmd[@]}" > "${out}.log" 2>&1 || { echo "[phase] seed=$s FAILED (see ${out}.log)"; exit 1; }
done
echo "[phase] ALL SEEDS DONE — output at $OUT_ROOT/s*_final/per_episode.parquet"
