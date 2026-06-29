#!/usr/bin/env bash
# scripts/agents/crl_flowbc/phase.sh — phase-probe over all CRL+FlowBC seeds.
#
# For each sdNNN_* under RESULTS_ROOT, picks the highest-step params_*.pkl
# and invokes scripts/probes/phase_probe_crl.py with the paper's 10-rollout
# deterministic protocol.
USAGE="usage: bash scripts/agents/crl_flowbc/phase.sh
  knobs (env vars):
    SEEDS='0 1 2 3 4 5 6 7 8 9'   seeds to process (default all 10)
    EPISODES=10                    rollouts per (task, scenario)
    SCENARIOS=S0,S1,S2             counterfactual scenarios (M1+M2)
    OUT_ROOT=analysis/probes/phase_probe_crl
    RESULTS_ROOT=results/factored-fb-crl-flowbc
    JAX_PY=.venv-jax-cpu/bin/python
    DRY_RUN=1                      preview commands only"
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
_maybe_help "$@"

SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
EPISODES="${EPISODES:-10}"
SCENARIOS="${SCENARIOS:-S0,S1,S2}"
OUT_ROOT="${OUT_ROOT:-analysis/probes/phase_probe_crl}"
RESULTS_ROOT="${RESULTS_ROOT:-results/factored-fb-crl-flowbc}"
JAX_PY="${JAX_PY:-.venv-jax-cpu/bin/python}"

mkdir -p "$OUT_ROOT"
for s in $SEEDS; do
    sdir=$(ls -d "${RESULTS_ROOT}"/sd$(printf '%03d' "$s")_* 2>/dev/null | head -1)
    if [[ -z "$sdir" ]]; then
        echo "[phase] seed=$s: NO RESULTS DIR — skipping"; continue
    fi
    # last step = highest-numbered params_*.pkl
    step=$(ls "$sdir"/params_*.pkl 2>/dev/null \
           | sed 's/.*params_//;s/\.pkl//' | sort -n | tail -1)
    if [[ -z "$step" ]]; then
        echo "[phase] seed=$s: NO CHECKPOINT in $sdir — skipping"; continue
    fi
    out="${OUT_ROOT}/s${s}_final"
    cmd=("$JAX_PY" scripts/probes/phase_probe_crl.py
         --checkpoint-dir "$sdir" --checkpoint-step "$step"
         --seed "$s" --out-dir "$out" --episodes "$EPISODES"
         --scenarios "$SCENARIOS")
    echo "[phase] seed=$s step=$step out=$out"
    if _dry_run_preview "${cmd[@]}"; then continue; fi
    "${cmd[@]}" || { echo "[phase] seed=$s FAILED"; exit 1; }
done
echo "[phase] ALL SEEDS DONE — output at $OUT_ROOT/s*_final/per_episode.parquet"
