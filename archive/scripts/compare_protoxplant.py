#!/usr/bin/env python3
"""Step-aligned comparison: our proto-transplant run vs the code-matched
PSM-orthohi reference (task2 success), seeds 0/5/7."""
import csv, glob, json, os, sys

G = "/var/local/amsks/exp/PSMFLows"
REF = json.load(open("/var/local/amsks/exp/ref_orthohi_task2_curves.json"))
SEEDS = [0, 5, 7]


def our_curve(seed):
    dirs = sorted(glob.glob(f"{G}/psm_protoxplant_*/sd{seed:03d}_*"))
    if not dirs:
        return {}, 0
    rows = {}
    last = 0
    for r in csv.DictReader(open(f"{dirs[-1]}/eval.csv")):
        try:
            s = int(float(r["step"])); rows[s] = float(r["evaluation/success"]); last = max(last, s)
        except (KeyError, ValueError):
            pass
    return rows, last


def val_at(rows, step):
    """last logged value at or before `step` (rows: {step: val})."""
    ok = [s for s in rows if s <= step]
    return rows[max(ok)] if ok else None


def main():
    print("=== proto-transplant run vs code-matched PSM-orthohi reference (task2) ===\n")
    all_done = True
    for seed in SEEDS:
        ours, last = our_curve(seed)
        refrows = {st: v for st, v in REF[str(seed)]}
        if last < 500000:
            all_done = False
        opk = max(ours.values()) if ours else float("nan")
        opk_s = (next(s for s, v in ours.items() if v == opk) // 1000) if ours else 0
        rpk = max(v for _, v in REF[str(seed)]); rpk_s = next(s for s, v in REF[str(seed)] if v == rpk) // 1000
        print(f"seed {seed}  (ours -> {last//1000}k):")
        line_o, line_r = [], []
        for st in sorted(refrows):
            if st > 500000:
                break
            ov = val_at(ours, st)
            line_o.append(f"{st//1000}k:{'  -  ' if ov is None else f'{ov:.2f}'}")
            line_r.append(f"{st//1000}k:{refrows[st]:.2f}")
        print("   ours:", "  ".join(line_o))
        print("   ref :", "  ".join(line_r))
        print(f"   peak-so-far ours {opk:.2f}@{opk_s}k  |  ref peak {rpk:.2f}@{rpk_s}k (late)\n")
    print(f"[all seeds reached 500k: {all_done}]")
    return all_done


if __name__ == "__main__":
    sys.exit(0 if main() else 2)
