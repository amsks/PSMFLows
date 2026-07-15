#!/usr/bin/env python3
"""Per-seed comparison: our multi-seed JAX FB run vs the reference FB (fb_flowbc,
ortho1000, lrb1e-4) task2 curves cached from wandb amsks/factored-fb.

Reference cache: /var/local/amsks/exp/ref_fb_ortho1000_task2_curves.json
(built by pulling `eval/reward/cube-single-play-singletask-task2-v0/success`).
Group name from argv[1] or /var/local/amsks/exp/fb_ortho1000_group.txt.

Peaks/means are computed WITHIN the step window both sides share (default <=500k),
NOT against the reference's global peak — the reference blooms late (peaks 500-750k),
so a global-peak comparison against a 500k run is apples-to-oranges.
"""
import csv, glob, json, os, sys

REF = json.load(open("/var/local/amsks/exp/ref_fb_ortho1000_task2_curves.json"))
GROUP = (sys.argv[1] if len(sys.argv) > 1
         else open("/var/local/amsks/exp/fb_ortho1000_group.txt").read().strip())
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 500000
BASE = "/var/local/amsks/exp/PSMFLows"


def our_curve(seed):
    dirs = sorted(glob.glob(f"{BASE}/{GROUP}/sd{seed:03d}_*"))
    if not dirs:
        return {}
    rows = {}
    for r in csv.DictReader(open(f"{dirs[-1]}/eval.csv")):
        try:
            s = int(float(r["step"]))
            if s <= CAP:
                rows[s] = float(r["evaluation/success"])
        except (KeyError, ValueError):
            pass
    return rows


def ref_at(seed, step):
    ok = [v for st, v in REF.get(str(seed), []) if st <= step]
    return ok[-1] if ok else None


def ref_inwindow(seed):
    return [(st, v) for st, v in REF.get(str(seed), []) if st <= CAP]


def main():
    print(f"group: {GROUP}   (reference: FB fb_flowbc ortho1000 lrb1e-4, task2, <= {CAP//1000}k)\n")
    seeds = sorted(int(s) for s in REF)  # reference seeds (1..10); intersect with ours
    all_pairs, summ = [], []
    for seed in seeds:
        ours = our_curve(seed)
        if not ours:
            continue
        steps = sorted(ours)
        pairs = [(s, ours[s], ref_at(seed, s)) for s in steps if ref_at(seed, s) is not None]
        if not pairs:
            continue
        print(f"seed {seed}  (step-matched, 0-{CAP//1000}k)")
        print("   step:", "  ".join(f"{s//1000:>4}k" for s, _, _ in pairs))
        print("   ours:", "  ".join(f"{o:>5.2f}" for _, o, _ in pairs))
        print("   ref :", "  ".join(f"{r:>5.2f}" for _, _, r in pairs))
        o_pk = max(o for _, o, _ in pairs)
        r_in = ref_inwindow(seed)
        r_pk = max(v for _, v in r_in) if r_in else float("nan")
        o_mu = sum(o for _, o, _ in pairs) / len(pairs)
        r_mu = sum(r for _, _, r in pairs) / len(pairs)
        print(f"   in-window peak: ours {o_pk:.2f}  ref {r_pk:.2f}  |  mean: ours {o_mu:.3f}  ref {r_mu:.3f}\n")
        summ.append((seed, o_mu, r_mu, o_pk, r_pk))
        all_pairs.extend(pairs)

    if not summ:
        print("(no matching seeds with eval.csv yet)")
        return
    print("=" * 60)
    print(f"{'seed':>4} | {'ours mean':>9} {'ref mean':>8} | {'ours pk':>7} {'ref pk':>6}")
    print("-" * 60)
    for seed, o_mu, r_mu, o_pk, r_pk in summ:
        print(f"{seed:>4} | {o_mu:>9.3f} {r_mu:>8.3f} | {o_pk:>7.2f} {r_pk:>6.2f}")
    o_all = sum(o for _, o, _ in all_pairs) / len(all_pairs)
    r_all = sum(r for _, _, r in all_pairs) / len(all_pairs)
    print(f"\npooled step-matched mean (<= {CAP//1000}k):  ours {o_all:.3f}  vs  ref {r_all:.3f}  (diff {o_all-r_all:+.3f})")
    print("NOTE: reference blooms late (global peaks 500-750k); this is the <=cap in-window view.")


if __name__ == "__main__":
    main()
