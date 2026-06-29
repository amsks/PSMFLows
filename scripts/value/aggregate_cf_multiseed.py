#!/usr/bin/env python
"""scripts/value/aggregate_cf_multiseed.py — aggregate the counterfactual value probe
across the 10 training seeds.

Reads analysis/value/repsep/cf_value_<method>_ms*.json (one per seed) and reports, per
method, the across-seed IQM and 95% bootstrap CI half-width of (i) the
counterfactual value rank-AUC and (ii) the counterfactual success rate -- the
same IQM +/- half-width convention as the per-task success table.

Writes analysis/value/repsep/cf_value_multiseed.json and prints a markdown table.
Run under .venv.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as _st

REPO = Path(__file__).resolve().parent.parent.parent
REPSEP = REPO / "analysis" / "value" / "repsep"
METHODS = ["gciql", "crl", "fb", "rldp"]
LABEL = {"gciql": "GCIQL", "crl": "CRL", "fb": "FB", "rldp": "RLDP"}


def _iqm_ci(samples, reps=5000, seed=0):
    x = np.asarray(samples, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    iqm = lambda a: _st.trim_mean(a, 0.25)
    rng = np.random.default_rng(seed)
    boot = np.array([iqm(rng.choice(x, len(x), replace=True)) for _ in range(reps)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(iqm(x)), float((hi - lo) / 2)


def main() -> int:
    out = {}
    rows = []
    for m in METHODS:
        files = sorted(glob.glob(str(REPSEP / f"cf_value_{m}_ms*.json")))
        if not files:
            print(f"[warn] no per-seed files for {m}", file=sys.stderr)
            continue
        auc_per_seed, sr_per_seed, per_task = [], [], []
        for f in files:
            d = json.loads(Path(f).read_text())
            auc_per_seed.append(d["cf_value_auc_mean"])
            sr_per_seed.append(float(np.mean(d["cf_success_rate"])))
            per_task.append(d["cf_value_auc_per_task"])
        auc_iqm, auc_hw = _iqm_ci(auc_per_seed)
        sr_iqm, sr_hw = _iqm_ci(sr_per_seed)
        out[m] = {
            "n_seeds": len(files),
            "auc_per_seed": auc_per_seed,
            "auc_iqm": auc_iqm, "auc_ci_halfwidth": auc_hw,
            "auc_mean": float(np.mean(auc_per_seed)), "auc_std": float(np.std(auc_per_seed, ddof=1)),
            "auc_min": float(np.min(auc_per_seed)), "auc_max": float(np.max(auc_per_seed)),
            "cf_success_iqm": sr_iqm, "cf_success_ci_halfwidth": sr_hw,
            "per_task_auc_per_seed": per_task,
        }
        rows.append((m, len(files), auc_iqm, auc_hw, sr_iqm, sr_hw,
                     np.min(auc_per_seed), np.max(auc_per_seed)))

    (REPSEP / "cf_value_multiseed.json").write_text(json.dumps(out, indent=2))

    print(f"\n{'Method':6} {'seeds':>5} {'AUC (IQM±CI)':>16} {'cf-succ (IQM±CI)':>18} {'AUC range':>16}")
    md = ["| Method | counterfactual value-AUC | cf. success rate |",
          "|---|---|---|"]
    for m, n, a, ah, s, sh, amin, amax in rows:
        print(f"{LABEL[m]:6} {n:>5} {a:>8.3f} ± {ah:.3f}   {s:>8.3f} ± {sh:.3f}   [{amin:.3f}, {amax:.3f}]")
        md.append(f"| {LABEL[m]} | ${a:.2f} \\pm {ah:.2f}$ | ${s:.2f} \\pm {sh:.2f}$ |")
    (REPSEP / "cf_value_multiseed_table.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {REPSEP/'cf_value_multiseed.json'} and cf_value_multiseed_table.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
