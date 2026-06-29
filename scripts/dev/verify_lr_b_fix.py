"""scripts/dev/verify_lr_b_fix.py — Confirm lr_b <= 1e-5 prevents FB collapse.

Same contract as scripts/dev/verify_ortho_fix.py, but isolates a different
intervention. The prior sweep at amsks/factored-fb used the broken default
(ortho_coef=1.0, clip_grad_norm=0, lr_b=1e-4) and every run collapsed. td_jepa
sweeps `lr_b ∈ {1e-4, 1e-5}` alongside `ortho_coef ∈ {100, 1000}`, so the
alternative hypothesis is: maybe `lr_b = 1e-5` alone slows B's drift enough to
preserve orthonormality and prevent collapse, even at ortho_coef = 1.0.

This script holds the *other* broken settings fixed (ortho_coef=1.0,
clip_grad_norm=0) and only varies lr_b — so a pass means lr_b is independently
sufficient to fix collapse; a fail means lr_b alone is not enough and you need
the ortho_coef bump (run `make verify-ortho-fix` for that).

Collapse signature (asserted not present by step 50k):

  - train/M1                stays below 40
  - train/q (abs)           stays below 100
  - train/orth_loss_offdiag stays below 60

Examples
--------
    python scripts/dev/verify_lr_b_fix.py                       # lr_b=1e-5, ortho=1.0
    python scripts/dev/verify_lr_b_fix.py --lr-b 1.0e-4         # reproduce collapse
    python scripts/dev/verify_lr_b_fix.py --device cpu \
        --num-train-steps 10000 --log-every 500             # quick local check
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

METRIC_RE = re.compile(r"\[step\s+(\d+)\]\s+(.+)")
KV_RE = re.compile(r"(\S+?)=(\S+)")

THRESHOLDS = {
    "train/M1":                ("max",     40.0),
    "train/q":                 ("abs_max", 100.0),
    "train/orth_loss_offdiag": ("max",     60.0),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", default="antmaze_medium")
    # Hold the *other* broken-sweep knobs fixed so this isolates lr_b.
    ap.add_argument("--ortho-coef", type=float, default=1.0,
                    help="Held at the broken default (1.0) to isolate lr_b's effect.")
    ap.add_argument("--clip-grad-norm", type=float, default=0.0,
                    help="Held at the broken default (0.0) to isolate lr_b's effect.")
    ap.add_argument("--lr-b", type=float, default=1.0e-5,
                    help="The candidate fix (default 1e-5; pass 1e-4 to reproduce collapse).")
    ap.add_argument("--num-train-steps", type=int, default=50_000)
    ap.add_argument("--log-every", type=int, default=2_500)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-wandb", action="store_true",
                    help="Disable wandb logging (default: enabled, group=verify-lr-b-fix).")
    ap.add_argument("--wandb-group", default="verify-lr-b-fix",
                    help="wandb group for this verification run (default: verify-lr-b-fix).")
    return ap.parse_args()


def _parse_value(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def main() -> None:
    args = parse_args()

    overrides = [
        f"domain={args.domain}",
        f"ortho_coef={args.ortho_coef}",
        f"lr_b={args.lr_b}",
        f"clip_grad_norm={args.clip_grad_norm}",
        f"num_train_steps={args.num_train_steps}",
        f"log_every={args.log_every}",
        f"eval_every={args.num_train_steps + 1}",
        f"save_every={args.num_train_steps + 1}",
        f"batch_size={args.batch_size}",
        f"device={args.device}",
        f"seed={args.seed}",
        f"use_wandb={'false' if args.no_wandb else 'true'}",
        f"wandb_entity=amsks",
        f"wandb_group={args.wandb_group}",
        f"wandb_tags=[verify-lr-b-fix]",
    ]
    cmd = [sys.executable, "-u", "train.py"] + overrides
    print(f"[verify-lr_b] running: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    series: dict[str, list[tuple[int, float]]] = {k: [] for k in THRESHOLDS}
    last_step = -1

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            print(line, flush=True)
            m = METRIC_RE.match(line)
            if not m:
                continue
            step = int(m.group(1))
            last_step = step
            tail = m.group(2)
            for k, v in KV_RE.findall(tail):
                if k not in THRESHOLDS:
                    continue
                val = _parse_value(v)
                if val is not None:
                    series[k].append((step, val))
        rc = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        raise

    print()
    print("[verify-lr_b] ============= verdict =============")
    print(f"[verify-lr_b] last step parsed: {last_step}")
    print(f"[verify-lr_b] train.py exit code: {rc}")
    print(f"[verify-lr_b] overrides: ortho_coef={args.ortho_coef}, lr_b={args.lr_b}, "
          f"clip_grad_norm={args.clip_grad_norm}")

    if rc != 0:
        print(f"[verify-lr_b] train.py crashed (exit={rc}) — collapse may have caused it")

    failed = False
    for key, (mode, limit) in THRESHOLDS.items():
        pts = series[key]
        if not pts:
            print(f"  ?? {key:30s}  NO SAMPLES (parser bug or no logging)")
            failed = True
            continue
        vals = [v for _, v in pts]
        if mode == "max":
            peak = max(vals)
        else:  # abs_max
            peak = max(abs(v) for v in vals)
        ok = peak < limit
        flag = "OK  " if ok else "FAIL"
        print(f"  {flag} {key:30s}  peak={peak:8.2f}  limit={limit:8.2f}  "
              f"({len(vals)} samples)")
        if not ok:
            failed = True

    if failed or rc != 0:
        print()
        print("[verify-lr_b] COLLAPSE SIGNATURE DETECTED — lr_b alone did not fix it.")
        print("[verify-lr_b] Next step: try `make verify-ortho-fix` (ortho_coef=100).")
        sys.exit(1)

    print()
    print("[verify-lr_b] No collapse signature — lr_b={:.0e} is independently sufficient."
          .format(args.lr_b))
    sys.exit(0)


if __name__ == "__main__":
    main()
