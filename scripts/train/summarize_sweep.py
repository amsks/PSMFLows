"""scripts/train/summarize_sweep.py — aggregate a grid_* log dir into a winner table.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/train/summarize_sweep.py \
      logs/grid_<TS> --knob-a ortho_coef --knob-b lr_phi --out analysis/misc/sweep/psm_state
Run logs are named <base>__<knobA><val>__<knobB><val>__s<seed>.log, where value
tags replace '.' with 'p' (e.g. 0.1 -> 0p1); this reverses that.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from scripts.train.sweep_lib import aggregate_grid, parse_final_success, pick_winner


def _untag(t: str) -> str:
    return t.replace("p", ".") if re.fullmatch(r"\d+p\d+", t) else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--knob-a", required=True)
    ap.add_argument("--knob-b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pat = re.compile(rf"{args.knob_a}([0-9pe.-]+)__{args.knob_b}([0-9pe.-]+)__s(\d+)\.log$")
    rows = defaultdict(list)
    for log in sorted(Path(args.log_dir).glob("*.log")):
        m = pat.search(log.name)
        if not m:
            continue
        a, b = _untag(m.group(1)), _untag(m.group(2))
        rows[(a, b)].append(parse_final_success(str(log)))

    agg = aggregate_grid(rows)
    winner = pick_winner(agg) if agg else None

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Sweep: {args.knob_a} x {args.knob_b}  ({args.log_dir})\n",
             f"| {args.knob_a} | {args.knob_b} | mean success | std | n |",
             "|---|---|---|---|---|"]

    def _mean_key(kv):
        m = kv[1]["mean"]
        return -1.0 if m != m else m  # NaN sorts last

    for (a, b), s in sorted(agg.items(), key=_mean_key, reverse=True):
        star = " **<-- winner**" if (a, b) == winner else ""
        lines.append(f"| {a} | {b} | {s['mean']:.3f} | {s['std']:.3f} | {s['n']} |{star}")
    lines.append(f"\n**Winner:** {args.knob_a}={winner[0]}, {args.knob_b}={winner[1]}" if winner
                 else "\n(no runs found)")
    out.with_suffix(".md").write_text("\n".join(lines))
    out.with_suffix(".json").write_text(json.dumps(
        {f"{a}|{b}": s for (a, b), s in agg.items()} | {"winner": list(winner) if winner else None}, indent=2))
    print("\n".join(lines))
    print(f"\nwrote {out}.md / {out}.json")


if __name__ == "__main__":
    main()
