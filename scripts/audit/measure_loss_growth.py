#!/usr/bin/env python3
"""Fit the exponential growth rate of `psm_loss` from a run's own `train.csv`.

Read-only, CPU-only, no jax: it reads `train.csv` and `flags.json` from finished or
in-flight Stage-C runs and reports, per run,

  steps_per_decade      1 / slope of log10(psm_loss) vs step, least squares over a window
  q_level               |Q| = psi_q_spread / psi_q_spread_rel at the window end
  q_growth_decades      log10(|Q|_end / |Q|_start) over the same window
  orth_depart_step      first step at which orth_offdiag > 2x its own 10-90k baseline
  index_spread_end      psi_q_index_spread_rel at the window end
  psm_diag_end          the diagonal (linear) term at the window end

and then fits the Part-A loop-gain model across runs:

  d log10(psm_loss) / d step  =  log10( gamma * (1 + c) ) / T_eff

with two free parameters shared by every arm at the same pessimism setting: the
per-backup relative pessimism bias `c` = kappa * E|M1 - M2| / |M| and the effective
backup period `T_eff` in gradient steps. A single (c, T_eff) that reproduces the measured
rate at three discounts is the empirical content of the claim that the exact-min ensemble
target, not the discount, is what makes the backup non-contractive.

Usage:
    .venv/bin/python scripts/audit/measure_loss_growth.py \
        --exp $PSM_DATA/exp/PSMFLows --report_out $PSM_DATA/logs/audit_measure_loss/growth.json
"""

import argparse
import csv
import glob
import json
import math
import os


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    if not rows:
        return out
    for k in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (TypeError, ValueError):
                vals.append(float("nan"))
        out[k] = vals
    return out


def lstsq_slope(xs, ys):
    """Least-squares slope of ys on xs; None when fewer than three finite pairs."""
    pts = [(x, y) for x, y in zip(xs, ys)
           if math.isfinite(x) and math.isfinite(y)]
    if len(pts) < 3:
        return None, None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    if sxx == 0:
        return None, None
    slope = sxy / sxx
    # R^2
    syy = sum((p[1] - my) ** 2 for p in pts)
    r2 = (sxy ** 2) / (sxx * syy) if syy > 0 else float("nan")
    return slope, r2


def at_step(steps, vals, target):
    """Value of `vals` at the logged step closest to `target`."""
    best, bv = None, None
    for s, v in zip(steps, vals):
        if not math.isfinite(s):
            continue
        d = abs(s - target)
        if best is None or d < best:
            best, bv = d, v
    return bv


def analyse_run(run_dir, lo, hi):
    csv_path = os.path.join(run_dir, "train.csv")
    if not os.path.exists(csv_path):
        return None
    d = read_csv(csv_path)
    if "step" not in d or "training/psm_loss" not in d:
        return None
    steps = d["step"]
    psm = d["training/psm_loss"]
    hi_eff = min(hi, max(s for s in steps if math.isfinite(s)))
    win = [(s, p) for s, p in zip(steps, psm) if lo <= s <= hi_eff and p > 0]
    slope, r2 = lstsq_slope([s for s, _ in win], [math.log10(p) for _, p in win])
    spread = d.get("training/psi_q_spread", [])
    spread_rel = d.get("training/psi_q_spread_rel", [])
    q = [a / b if (b and math.isfinite(a) and math.isfinite(b) and b != 0) else float("nan")
         for a, b in zip(spread, spread_rel)] if spread and spread_rel else []

    # orth_offdiag departure: first step past `lo` where it exceeds 2x the 10-90k baseline
    off = d.get("training/orth_offdiag", [])
    depart = None
    if off:
        base = [v for s, v in zip(steps, off) if 10000 <= s <= 90000 and math.isfinite(v)]
        if base:
            b = sum(base) / len(base)
            for s, v in zip(steps, off):
                if s > 90000 and math.isfinite(v) and v > 2 * b:
                    depart = s
                    break

    flags_path = os.path.join(run_dir, "flags.json")
    fl = {}
    if os.path.exists(flags_path):
        with open(flags_path) as f:
            fl = json.load(f)
    ag = fl.get("agent", fl)

    def cfg(k, default=None):
        v = ag.get(k, default)
        return v

    q_growth = None
    if q:
        q0, q1 = at_step(steps, q, lo), at_step(steps, q, hi_eff)
        if q0 and q1 and q0 > 0 and q1 > 0:
            q_growth = math.log10(q1 / q0)

    return dict(
        run=os.path.basename(run_dir),
        group=os.path.basename(os.path.dirname(run_dir)),
        env=fl.get("env_name"),
        discount=cfg("discount"),
        pessimism_penalty=cfg("pessimism_penalty"),
        ortho_coef=cfg("ortho_coef"),
        ortho_mode=cfg("ortho_mode"),
        psi_bound=cfg("psi_bound"),
        lr_sf=cfg("lr_sf"), lr_phi=cfg("lr_phi"), tau=cfg("tau"),
        num_parallel=cfg("num_parallel"), psi_form=cfg("psi_form"),
        window=[lo, hi_eff],
        n_points=len(win),
        decades_per_step=slope,
        steps_per_decade=(1.0 / slope) if slope and slope > 0 else None,
        fit_r2=r2,
        psm_loss_lo=at_step(steps, psm, lo),
        psm_loss_hi=at_step(steps, psm, hi_eff),
        psm_diag_hi=at_step(steps, d.get("training/psm_diag", []), hi_eff),
        psm_offdiag_hi=at_step(steps, d.get("training/psm_offdiag", []), hi_eff),
        q_level_hi=at_step(steps, q, hi_eff) if q else None,
        q_growth_decades=q_growth,
        orth_offdiag_hi=at_step(steps, off, hi_eff) if off else None,
        orth_depart_step=depart,
        index_spread_rel_hi=at_step(steps, d.get("training/psi_q_index_spread_rel", []), hi_eff),
        q_spread_rel_hi=at_step(steps, d.get("training/psi_q_spread_rel", []), hi_eff),
        w_enc_spread_hi=at_step(steps, d.get("training/w_enc_spread", []), hi_eff),
        psi_absmean_hi=at_step(steps, d.get("training/psi_absmean", []), hi_eff),
        td_target_absmean_hi=at_step(steps, d.get("training/td_target_absmean", []), hi_eff),
        psi_bound_frac_hi=at_step(steps, d.get("training/psi_bound_frac", []), hi_eff),
        max_step=hi_eff,
    )


def fit_loop_gain(rows):
    """Solve d log10(psm)/dstep = log10(gamma*(1+c)) / T_eff for (c, T_eff).

    Grid search on c; T_eff is then the least-squares scale that maps the predicted
    per-backup log-gain onto the measured per-step rate. Runs whose measured rate is
    non-positive (bounded psm_loss) contribute a one-sided constraint only: they are
    reported as `bounded` and must satisfy gamma*(1+c) <= 1 for the fit to be consistent.
    """
    use = [r for r in rows if r.get("decades_per_step") is not None and r.get("discount")]
    if not use:
        return None
    # A run counts as DIVERGENT when its fitted psm_loss growth is faster than one decade
    # per BOUNDED_SPD steps; slower than that is indistinguishable from a bounded trace over
    # a 500k budget and enters only as the one-sided constraint gamma*(1+c) <= 1.
    BOUNDED_SPD = 1.0e6
    div = [r for r in use if r["decades_per_step"] > 1.0 / BOUNDED_SPD]
    bnd = [r for r in use if r["decades_per_step"] <= 1.0 / BOUNDED_SPD]
    best = None
    c = 1e-5
    while c < 0.2:
        gains = [math.log10(float(r["discount"]) * (1.0 + c)) for r in div]
        ok = bool(gains) and all(g > 0 for g in gains) and all(
            math.log10(float(r["discount"]) * (1.0 + c)) <= 0 for r in bnd)
        if ok:
            obs = [r["decades_per_step"] for r in div]
            # T_eff minimising sum ((gain/T)/obs - 1)^2  ->  T = sum(g^2/o^2)/sum(g/o)
            num = sum(g * g / (o * o) for g, o in zip(gains, obs))
            den = sum(g / o for g, o in zip(gains, obs))
            if den > 0:
                T = num / den
                err = sum((g / T / o - 1.0) ** 2 for g, o in zip(gains, obs))
                if best is None or err < best[0]:
                    best = (err, c, T, err, 0.0)
        c *= 1.002
    if best is None:
        return None
    _, c, T, err, pen = best
    out = dict(c=c, kappa_times_rel_disagreement=c, T_eff_steps=T,
               rel_sq_error=err, sign_violations=pen,
               gamma_critical=1.0 / (1.0 + c),
               note="rate = log10(gamma*(1+c)) / T_eff; gamma_critical is where the "
                    "pessimism-inflated backup stops contracting")
    out["per_run"] = []
    for r in use:
        g = float(r["discount"])
        gain = math.log10(g * (1.0 + c))
        out["per_run"].append(dict(
            group=r["group"], run=r["run"], discount=g,
            measured_steps_per_decade=r["steps_per_decade"],
            predicted_steps_per_decade=(T / gain) if gain > 0 else None,
            predicted_bounded=(gain <= 0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="$PSM_DATA/exp/PSMFLows")
    ap.add_argument("--groups", default="", help="comma-separated group names; empty = all")
    ap.add_argument("--fit_groups", default="",
                    help="comma-separated groups the loop-gain fit is restricted to "
                         "(default: every group listed). Use it to keep stabilised arms "
                         "(psi_bound != none) out of the fit -- they change the loop, not "
                         "the discount.")
    ap.add_argument("--lo", type=float, default=50000.0)
    ap.add_argument("--hi", type=float, default=500000.0)
    ap.add_argument("--report_out", default="")
    args = ap.parse_args()

    groups = [g for g in args.groups.split(",") if g] or sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(args.exp, "*")) if os.path.isdir(p))
    rows = []
    for g in groups:
        for run in sorted(glob.glob(os.path.join(args.exp, g, "sd*"))):
            r = analyse_run(run, args.lo, args.hi)
            if r:
                rows.append(r)
    fit_groups = [g for g in args.fit_groups.split(",") if g]
    fit_rows = [r for r in rows if r["group"] in fit_groups] if fit_groups else rows
    report = dict(exp=args.exp, window=[args.lo, args.hi],
                  fit_groups=fit_groups or groups, runs=rows,
                  loop_gain_fit=fit_loop_gain(fit_rows))
    txt = json.dumps(report, indent=2)
    print(txt)
    if args.report_out:
        os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
        with open(args.report_out, "w") as f:
            f.write(txt)
        print(f"report -> {args.report_out}")


if __name__ == "__main__":
    main()
