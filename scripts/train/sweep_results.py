"""scripts/train/sweep_results.py — extract eval/success across the running sweep
and emit a markdown table similar to expected_results.md.

For each run started in the last 36h on the amsks/factored-fb project, group
by (domain, ortho_coef, lr_b) and pull the LATEST non-null
`eval/reward/eval/success` per run. Report mean ± std (× 100, as percentage)
across seeds for each (domain, hyper) cell, plus the step at which the value
was measured for transparency.

Usage:
    python scripts/train/sweep_results.py
    python scripts/train/sweep_results.py --hours 36 --out results/sweep_2026-05-14.md
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import wandb


# (domain, ortho_coef, lr_b)  →  list of (seed, step, success, reward)
RunRecord = Tuple[int, int, float, float]


DOMAIN_LABELS = {
    "antmaze-medium-navigate-v0": "antmaze-medium",
    "cube-single-play-v0":        "cube-single",
}

HYPER_ORDER = [
    ("100",  "1e-4"),
    ("100",  "1e-5"),
    ("1000", "1e-4"),
    ("1000", "1e-5"),
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default="amsks/factored-fb")
    ap.add_argument("--hours", type=int, default=36, help="Look-back window")
    ap.add_argument("--out", default=None, help="Optional markdown output path")
    return ap.parse_args()


def fmt_lr(lr: float) -> str:
    # 1e-4 -> "1e-4", 1e-5 -> "1e-5"
    s = f"{lr:.0e}".replace("e-0", "e-")
    return s


def fetch_records(project: str, hours: int) -> Dict[Tuple[str, str, str], List[RunRecord]]:
    api = wandb.Api()
    runs = list(api.runs(project, order="-created_at", per_page=300))[:300]
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    out: Dict[Tuple[str, str, str], List[RunRecord]] = defaultdict(list)

    for r in runs:
        c = (datetime.fromisoformat(r.created_at.replace("Z", "+00:00"))
             if isinstance(r.created_at, str) else r.created_at)
        if c < cutoff:
            continue
        cfg = r.config
        domain = cfg.get("domain")
        if domain not in DOMAIN_LABELS:
            continue
        ortho = str(int(cfg.get("ortho_coef")))
        lr_b = fmt_lr(float(cfg.get("lr_b")))
        seed = int(cfg.get("seed"))

        # Pull only the eval columns to keep it cheap
        keys = ["eval/reward/eval/success", "eval/reward/eval/reward"]
        try:
            hist = r.history(keys=keys, samples=2000, pandas=True)
        except Exception as e:
            print(f"  [warn] history fetch failed for {r.id}: {e}")
            continue
        if hist.empty:
            continue
        succ_col = "eval/reward/eval/success"
        rew_col = "eval/reward/eval/reward"
        if succ_col not in hist.columns:
            continue
        nn = hist[hist[succ_col].notna()]
        if nn.empty:
            continue
        last = nn.iloc[-1]
        step = int(last["_step"])
        success = float(last[succ_col])
        reward = float(last[rew_col]) if rew_col in hist.columns else float("nan")

        out[(domain, ortho, lr_b)].append((seed, step, success, reward))

    return out


def render_table(records: Dict[Tuple[str, str, str], List[RunRecord]]) -> str:
    lines: List[str] = []
    lines.append("# Sweep results (latest eval per run × 100)")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append("")

    # Header — one column per hyper combo, plus a "best" column
    header_cells = ["**Task**"] + [f"ortho={o}, lr_b={lr}" for (o, lr) in HYPER_ORDER]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| " + " | ".join([":---"] * len(header_cells)) + " |")

    for domain, label in DOMAIN_LABELS.items():
        cells = [f"**{label}**"]
        cell_vals = []
        for o, lr in HYPER_ORDER:
            recs = records.get((domain, o, lr), [])
            if not recs:
                cells.append("—")
                cell_vals.append(None)
                continue
            succ = np.array([r[2] for r in recs]) * 100.0
            mean = succ.mean()
            std = succ.std(ddof=1) if len(succ) > 1 else 0.0
            steps = [r[1] for r in recs]
            step_min, step_max = min(steps), max(steps)
            step_disp = f"{step_min // 1000}k" if step_min == step_max else f"{step_min // 1000}–{step_max // 1000}k"
            cells.append(f"{mean:.2f} ± {std:.2f} (n={len(succ)}, @{step_disp})")
            cell_vals.append(mean)

        # Bold the column with highest mean for this row
        best_idx = None
        if any(v is not None for v in cell_vals):
            best_idx = int(np.argmax([(v if v is not None else -1e9) for v in cell_vals]))
        if best_idx is not None:
            cells[1 + best_idx] = f"**{cells[1 + best_idx]}**"

        lines.append("| " + " | ".join(cells) + " |")

    # Per-seed dump for transparency
    lines.append("")
    lines.append("## Per-seed values (success × 100)")
    lines.append("")
    for (domain, o, lr), recs in sorted(records.items()):
        recs_sorted = sorted(recs, key=lambda r: r[0])
        seeds_str = ", ".join(f"s{s}={succ*100:.1f}@{step//1000}k"
                              for s, step, succ, _ in recs_sorted)
        lines.append(f"- **{DOMAIN_LABELS[domain]}** ortho={o} lr_b={lr}: {seeds_str}")
    lines.append("")

    return "\n".join(lines)


def main():
    args = parse_args()
    print(f"[sweep] fetching runs from {args.project} (last {args.hours}h)")
    records = fetch_records(args.project, args.hours)
    total_runs = sum(len(v) for v in records.values())
    print(f"[sweep] collected {total_runs} runs across {len(records)} hyper cells")
    table = render_table(records)
    print()
    print(table)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table)
        print(f"\n[sweep] written to {args.out}")


if __name__ == "__main__":
    main()
