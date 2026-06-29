#!/usr/bin/env bash
# scripts/agents/rldp_flowbc/phase.sh — phase-probe over all RLDP+FlowBC seeds.
#
# RLDP is a PyTorch FB-family flow-BC agent, so it reuses the FB phase-probe
# driver (scripts/probes/phase_probe.py, which calls agent.act(obs, z)) — unlike CRL,
# which needs the JAX driver. For each __sN run under RESULTS_ROOT it picks the
# final checkpoint and that run's saved .hydra/config.yaml.
#
# The rldp_flowbc checkpoint carries an extra `_predictor` head with no local
# implementation; evals.analysis.load_checkpoint loads the FB + flow-BC model
# subset and ignores it (see that function).
USAGE="usage: bash scripts/agents/rldp_flowbc/phase.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4 5 6 7 8 9'   seeds to process (default all 10)
    EPISODES=10                    rollouts per (task, scenario)
    SCENARIOS=S0,S1,S2             counterfactual scenarios (M1+M2)
    OUT_ROOT=analysis/probes/phase_probe_rldp
    RESULTS_ROOT=results/factored-fb-rldp-flowbc
    DATA_PATH=datasets             offline buffer root (cfg path is a cluster path)
    MUJOCO_GL=glfw                 GL backend (glfw on macOS; egl on Linux)
    PY=.venv/bin/python
    DRY_RUN=1                      preview commands only"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
EPISODES="${EPISODES:-10}"
SCENARIOS="${SCENARIOS:-S0,S1,S2}"
OUT_ROOT="${OUT_ROOT:-analysis/probes/phase_probe_rldp}"
RESULTS_ROOT="${RESULTS_ROOT:-results/factored-fb-rldp-flowbc}"
DATA_PATH="${DATA_PATH:-datasets}"
MUJOCO_GL="${MUJOCO_GL:-glfw}"
PY="${PY:-$REPO/.venv/bin/python}"

mkdir -p "$OUT_ROOT"
for s in $SEEDS; do
    sdir=$(ls -d "${RESULTS_ROOT}"/*__s${s} 2>/dev/null | head -1)
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
         --mujoco-gl "$MUJOCO_GL")
    echo "[phase] seed=$s out=$out"
    if _dry_run_preview "${cmd[@]}"; then continue; fi
    "${cmd[@]}" > "${out}.log" 2>&1 || { echo "[phase] seed=$s FAILED (see ${out}.log)"; exit 1; }
done
echo "[phase] ALL SEEDS DONE — output at $OUT_ROOT/s*_final/per_episode.parquet"
