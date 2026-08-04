"""Multi-seed, multi-agent comparison over eval.csv curves — mean +/- 95% CI and peaks.

Replaces the reference-JSON-hardwired compare_multiseed.py for the psmflow arc: compares
any number of agents (psmflow / psm / fb / fql ...) from their run directories, on the
protocol docs/reference_benchmarks.md prescribes — success on the shared eval grid,
mean +/- 95% CI across seeds at each step, and peak-over-grid per seed aggregated the
same way. No reference cache, no hardcoded paths.

Run:
  .venv/bin/python scripts/compare_zeroshot.py \
      fql='/data-local/amsks/PSMFLows/exp/PSMFLows/fqlbaseline_pointmaze_medium*/sd*' \
      psmflow='/data-local/amsks/PSMFLows/exp/PSMFLows/psmflow_pointmaze*/sd*' \
      [--metric evaluation/success] [--max-step 500000] [--out report.json]

Each LABEL=GLOB names one agent; the glob must match run dirs that contain eval.csv
(one per seed). Steps not shared by every seed of a label are dropped for that label;
the cross-label table uses each label's own grid (peaks are within --max-step).
"""
import argparse
import csv
import glob
import json
import math
import os

import numpy as np

# Two-sided 95% t critical values by seed count; falls back to scipy for larger n.
_T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}


def _t95(n):
    if n in _T95:
        return _T95[n]
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, n - 1))
    except ImportError:
        return 1.96


def mean_ci(vals):
    """(mean, half-width of the 95% CI) across seeds. CI is 0 for a single seed —
    a single-seed read is a point estimate and should be labeled as such, not dressed
    with a fake interval."""
    v = np.asarray(vals, np.float64)
    m = float(v.mean())
    if v.size < 2:
        return m, 0.0
    return m, float(_t95(v.size) * v.std(ddof=1) / math.sqrt(v.size))


def load_series(run_dir, metric):
    path = os.path.join(run_dir, "eval.csv")
    steps, vals = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get(metric) not in (None, ""):
                steps.append(int(float(row["step"])))
                vals.append(float(row[metric]))
    return dict(zip(steps, vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("groups", nargs="+", metavar="LABEL=GLOB")
    ap.add_argument("--metric", default="evaluation/success")
    ap.add_argument("--max-step", type=int, default=None,
                    help="compare only steps <= this (shared-window discipline)")
    ap.add_argument("--out", default=None, help="also write the report as JSON")
    args = ap.parse_args()

    report = {"metric": args.metric, "max_step": args.max_step, "agents": {}}
    for spec in args.groups:
        label, pattern = spec.split("=", 1)
        run_dirs = sorted(d for d in glob.glob(os.path.expanduser(pattern))
                          if os.path.isfile(os.path.join(d, "eval.csv")))
        assert run_dirs, f"{label}: no run dirs with eval.csv match {pattern!r}"
        series = {d: load_series(d, args.metric) for d in run_dirs}
        shared = sorted(set.intersection(*(set(s) for s in series.values())))
        if args.max_step:
            shared = [s for s in shared if s <= args.max_step]
        assert shared, f"{label}: seeds share no eval steps"
        curve = {}
        for step in shared:
            m, ci = mean_ci([series[d][step] for d in run_dirs])
            curve[step] = (round(m, 4), round(ci, 4))
        peaks = [max(v for s, v in series[d].items()
                     if not args.max_step or s <= args.max_step) for d in run_dirs]
        pm, pci = mean_ci(peaks)
        finals = [series[d][shared[-1]] for d in run_dirs]
        fm, fci = mean_ci(finals)
        report["agents"][label] = {
            "n_seeds": len(run_dirs),
            "runs": run_dirs,
            "curve": {str(k): v for k, v in curve.items()},
            "peak_per_seed": [round(p, 4) for p in peaks],
            "peak_mean_ci95": (round(pm, 4), round(pci, 4)),
            "final_step": shared[-1],
            "final_mean_ci95": (round(fm, 4), round(fci, 4)),
        }

    labels = list(report["agents"])
    print(f"metric: {args.metric}" + (f"   (steps <= {args.max_step})" if args.max_step else ""))
    print(f"{'step':>10}  " + "  ".join(f"{l:>20}" for l in labels))
    all_steps = sorted({int(s) for l in labels for s in report["agents"][l]["curve"]})
    for step in all_steps:
        cells = []
        for l in labels:
            c = report["agents"][l]["curve"].get(str(step))
            cells.append(f"{c[0]:.3f} +/- {c[1]:.3f}" if c else "-")
        print(f"{step:>10}  " + "  ".join(f"{c:>20}" for c in cells))
    print()
    for l in labels:
        a = report["agents"][l]
        pm, pci = a["peak_mean_ci95"]
        fm, fci = a["final_mean_ci95"]
        print(f"{l:>12}: n={a['n_seeds']}  peak {pm:.3f} +/- {pci:.3f}  "
              f"(per-seed {a['peak_per_seed']})  final@{a['final_step']} {fm:.3f} +/- {fci:.3f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
