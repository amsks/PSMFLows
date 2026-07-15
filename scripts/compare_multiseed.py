#!/usr/bin/env python3
"""Per-seed comparison: our multi-seed PSM run vs the code-matched PSM-orthohi
reference (task2 success), each eval aligned to the reference value at that step.

Reads the group name from /var/local/amsks/exp/multiseed_group.txt (or argv[1])."""
import csv, glob, json, os, sys

REF = json.load(open("/var/local/amsks/exp/ref_orthohi_task2_curves.json"))
GROUP = (sys.argv[1] if len(sys.argv) > 1
         else open("/var/local/amsks/exp/multiseed_group.txt").read().strip())
BASE = "/var/local/amsks/exp/PSMFLows"
SEEDS = [0, 1, 2, 3, 4, 5]


def our_curve(seed):
    dirs = sorted(glob.glob(f"{BASE}/{GROUP}/sd{seed:03d}_*"))
    if not dirs:
        return {}, 0
    rows, last = {}, 0
    for r in csv.DictReader(open(f"{dirs[-1]}/eval.csv")):
        try:
            s = int(float(r["step"])); rows[s] = float(r["evaluation/success"]); last = max(last, s)
        except (KeyError, ValueError):
            pass
    return rows, last


def ref_at(seed, step):
    ok = [v for st, v in REF.get(str(seed), []) if st <= step]
    return ok[-1] if ok else None


def main():
    print(f"group: {GROUP}\n")
    print("per-seed, ours vs reference (task2 success) at each eval checkpoint:\n")
    summ = []
    for seed in SEEDS:
        ours, last = our_curve(seed)
        if not ours:
            print(f"seed {seed}: (no eval.csv yet)\n"); continue
        opk = max(ours.values()); opk_s = next(s for s, v in ours.items() if v == opk) // 1000
        rpk = max(v for _, v in REF[str(seed)]); rpk_s = next(s for s, v in REF[str(seed)] if v == rpk) // 1000
        r100 = ref_at(seed, 100000); o100 = ours.get(100000)
        summ.append((seed, o100, r100, opk, opk_s, rpk, rpk_s, last))
        steps = sorted(ours)
        line_o = "  ".join(f"{s//1000}k:{ours[s]:.2f}" for s in steps)
        line_r = "  ".join(f"{s//1000}k:{(ref_at(seed,s) if ref_at(seed,s) is not None else float('nan')):.2f}"
                           for s in steps)
        print(f"seed {seed}  (ours -> {last//1000}k)")
        print(f"   ours: {line_o}")
        print(f"   ref : {line_r}")
        print(f"   peak: ours {opk:.2f}@{opk_s}k | ref {rpk:.2f}@{rpk_s}k (ref blooms late)\n")

    print("=" * 64)
    print(f"{'seed':>4} | {'ours@100k':>9} {'ref@100k':>8} | {'ours peak':>9} {'ref peak':>8}")
    print("-" * 64)
    for seed, o100, r100, opk, opk_s, rpk, rpk_s, last in summ:
        o = f"{o100:.2f}" if o100 is not None else "(<100k)"
        r = f"{r100:.2f}" if r100 is not None else "n/a"
        print(f"{seed:>4} | {o:>9} {r:>8} | {opk:>6.2f}@{opk_s:>3}k {rpk:>5.2f}@{rpk_s:>3}k")
    done = [s for s, *_ , last in summ if last >= 500000]
    print(f"\n[{len(done)}/{len(SEEDS)} seeds reached 500k]  ours-peak mean="
          f"{(sum(x[3] for x in summ)/len(summ)):.3f}  ref-peak mean={(sum(x[5] for x in summ)/len(summ)):.3f}" if summ else "")
    return len(done) == len(SEEDS)


if __name__ == "__main__":
    sys.exit(0 if main() else 2)
