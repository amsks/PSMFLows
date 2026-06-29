"""scripts/dev/verify_ortho_fix.py — Confirm ortho_coef >= 100 prevents FB collapse.

Spawns `python train.py` with the recommended antmaze-medium overrides for a
short horizon (default 50k steps), parses stdout for the metric trajectory,
and asserts the collapse signature does not appear:

  - train/M1                stays below 40   (collapse goes to 80-110)
  - train/q (abs)           stays below 100  (collapse goes to 400-550 = ||F||*||z||)
  - train/orth_loss_offdiag stays below 60   (collapse drifts upward to 80+)

Use this twice to compare: once with the fix (default), once with the broken
config (`--ortho-coef 1.0 --clip-grad-norm 0`) to confirm collapse on the bad
hyperparameters.

Examples
--------
    python scripts/dev/verify_ortho_fix.py                          # the fix
    python scripts/dev/verify_ortho_fix.py --ortho-coef 1.0 \
        --clip-grad-norm 0                                       # reproduce collapse
    python scripts/dev/verify_ortho_fix.py --device cpu \
        --num-train-steps 10000 --log-every 500                  # quick local check
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Match `[step    50000]  train/loss/fb=83.2600  ...`
METRIC_RE = re.compile(r"\[step\s+(\d+)\]\s+(.+)")
KV_RE = re.compile(r"(\S+?)=(\S+)")


# Collapse thresholds. Healthy training (ortho_coef >= 100) keeps these well
# below the limits; collapse (ortho_coef = 1) clearly exceeds them by 50k steps.
THRESHOLDS = {
    "train/M1":                ("max",     40.0),
    "train/q":                 ("abs_max", 100.0),
    "train/orth_loss_offdiag": ("max",     60.0),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", default="antmaze_medium")
    ap.add_argument("--ortho-coef", type=float, default=100.0)
    ap.add_argument("--lr-b", type=float, default=1.0e-4)
    ap.add_argument("--clip-grad-norm", type=float, default=1.0)
    ap.add_argument("--num-train-steps", type=int, default=50_000)
    ap.add_argument("--log-every", type=int, default=2_500)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-wandb", action="store_true",
                    help="Disable wandb logging (default: enabled, group=verify-ortho-fix).")
    ap.add_argument("--wandb-group", default="verify-ortho-fix",
                    help="wandb group for this verification run (default: verify-ortho-fix).")
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
        # Skip eval + ckpt entirely; we only need the train metric trace.
        f"eval_every={args.num_train_steps + 1}",
        f"save_every={args.num_train_steps + 1}",
        f"batch_size={args.batch_size}",
        f"device={args.device}",
        f"seed={args.seed}",
        f"use_wandb={'false' if args.no_wandb else 'true'}",
        f"wandb_entity=amsks",
        f"wandb_group={args.wandb_group}",
        f"wandb_tags=[verify-ortho-fix]",
    ]
    cmd = [sys.executable, "-u", "train.py"] + overrides
    print(f"[verify] running: {' '.join(cmd)}", flush=True)

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
    print("[verify] ============= verdict =============")
    print(f"[verify] last step parsed: {last_step}")
    print(f"[verify] train.py exit code: {rc}")
    print(f"[verify] overrides: ortho_coef={args.ortho_coef}, lr_b={args.lr_b}, "
          f"clip_grad_norm={args.clip_grad_norm}")

    if rc != 0:
        print(f"[verify] train.py crashed (exit={rc}) — collapse may have caused it")

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
        print("[verify] COLLAPSE SIGNATURE DETECTED — FB is diverging.")
        print("[verify] Symptoms above indicate F's norm is growing unboundedly")
        print("[verify] and/or B is losing orthonormality. Try ortho_coef in {100, 1000}.")
        sys.exit(1)

    print()
    print("[verify] No collapse signature — metrics stayed within healthy bounds.")
    sys.exit(0)


if __name__ == "__main__":
    main()
