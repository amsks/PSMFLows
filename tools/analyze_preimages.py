"""Preimage npz analyzer: everything Stage B wrote, characterized in one report.

Where validate_flow_inversion.py (D3) scores a fresh 256-row inversion against the gate,
this reads a FULL precomputed npz (all ~1M rows) and reports what Stage C will actually
train on: ESS / round-trip / typicality distributions, posterior geometry vs the N(0,I)
prior and the u_clip box, validity, and the correlations that tell proposal-mismatch
apart from sampling noise. Pure numpy — no GPU, no env, no checkpoint needed.

Run:  .venv/bin/python tools/analyze_preimages.py /path/to/preimages.npz
Writes <npz>.analysis.json and <npz>.analysis.png next to the npz (override with --out).
"""
import argparse
import json
import math
import os

import numpy as np

U_CLIP_DEFAULT = 3.0     # configs/agent/psmflow.yaml u_clip
ESS_GATE = 20.0          # D3 spec gate on mean final-step ESS
ROUNDTRIP_GATE = 0.1     # D3 spec gate on mean round-trip


def _stats(x):
    x = np.asarray(x, np.float64)
    return {
        "mean": round(float(np.mean(x)), 4),
        "median": round(float(np.median(x)), 4),
        "p01": round(float(np.percentile(x, 1)), 4),
        "p99": round(float(np.percentile(x, 99)), 4),
        "min": round(float(np.min(x)), 4),
        "max": round(float(np.max(x)), 4),
    }


def _corr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return None
    return round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 3)


def analyze(npz_path, u_clip=U_CLIP_DEFAULT):
    data = dict(np.load(npz_path))
    report = {"npz": os.path.abspath(npz_path)}

    meta_path = npz_path + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            report["meta"] = json.load(f)

    mean = data["noise_preimage_mean"]          # (N, K, A)
    cov = data["noise_preimage_cov"]            # (N, K, A, A)
    weights = data["noise_preimage_weights"]    # (N, K)
    n, k, d_a = mean.shape
    report["n"], report["n_components"], report["action_dim"] = n, k, d_a

    # --- validity: rows the repair path reset to the N(0,I) prior carry no information ---
    if "preimage_valid" in data:
        valid = data["preimage_valid"].astype(bool)
    else:  # pre-key npz: recompute the same conjunction main.py's loader uses
        valid = (np.isfinite(mean).all((1, 2)) & np.isfinite(cov).all((1, 2, 3))
                 & np.isfinite(weights).all(1))
    report["invalid"] = {"count": int((~valid).sum()),
                         "frac": round(float((~valid).mean()), 6)}

    # --- ESS of the stored (final-iterate) posterior ---
    ess = None
    if "preimage_ess" in data:
        ess = np.asarray(data["preimage_ess"], np.float64)
        report["ess"] = {**_stats(ess),
                         "frac_gt_gate": round(float((ess > ESS_GATE).mean()), 4),
                         "frac_zero": round(float((ess <= 0).mean()), 6),
                         "gate_mean_gt_20": bool(np.mean(ess) > ESS_GATE)}

    # --- point preimage: round-trip + typicality (the direct A2 test, Lemma 6.1) ---
    rt = None
    if "preimage_roundtrip" in data:
        rt = np.asarray(data["preimage_roundtrip"], np.float64)
        fin = rt[np.isfinite(rt)]
        report["roundtrip"] = {**_stats(fin),
                               "frac_gt_gate": round(float((fin > ROUNDTRIP_GATE).mean()), 6),
                               "gate_mean_lt_0.1": bool(np.mean(fin) < ROUNDTRIP_GATE)}
    sq = None
    if "noise_preimage_point" in data:
        pt = np.asarray(data["noise_preimage_point"], np.float64)
        sq = (pt ** 2).sum(-1)
        sqf = sq[np.isfinite(sq)]  # pre-repair npz can carry NaN point rows
        typ = {
            # Trimmed mean: the implicit-Euler inverse can blow up to huge FINITE values
            # (measured 1e59 on the alpha=1 cube npz) without reaching NaN, so a
            # finiteness-based validity mask does not catch these rows and a plain mean
            # is meaningless. frac_sq_gt_100 counts them: chi^2_{d_a} mass above 100 is
            # ~0 for any d_a here, so every such row is a diverged inverse.
            "mean_sq_norm_p999trim": round(float(sqf[sqf <= np.percentile(sqf, 99.9)].mean()), 4),
            "median_sq_norm": round(float(np.median(sqf)), 4),
            "expected_mean": d_a,
            "frac_sq_gt_100": round(float((sqf > 100).mean()), 6),
        }
        try:
            from scipy.stats import chi2, kstest
            lo, hi = chi2.ppf(0.005, d_a), chi2.ppf(0.995, d_a)
            typ["frac_in_99pct_band"] = round(float(((sqf >= lo) & (sqf <= hi)).mean()), 4)
            sub = sqf[:: max(1, n // 100000)]  # KS on ~100k rows is plenty
            ks = kstest(sub, lambda v: chi2.cdf(v, d_a))
            typ["ks_stat"], typ["ks_pvalue"] = round(float(ks.statistic), 4), float(ks.pvalue)
        except ImportError:
            typ["note"] = "scipy unavailable"
        report["typicality"] = typ

    # --- posterior geometry vs the prior and the u_clip box Stage C samples inside ---
    # Weighted mixture-mean per row; width = sqrt(trace/d) per component, weight-averaged
    # (prior width on this scale is exactly 1).
    mix_mean = (weights[..., None] * mean).sum(1)                       # (N, A)
    mu_norm = np.linalg.norm(mix_mean, axis=-1)
    tr = np.einsum("nkaa->nk", cov)
    width = (weights * np.sqrt(tr / d_a)).sum(1)
    chi_mean = math.sqrt(2) * math.exp(math.lgamma((d_a + 1) / 2) - math.lgamma(d_a / 2))
    report["posterior"] = {
        "mu_norm": _stats(mu_norm[valid]),
        "chi_typical_radius": round(float(chi_mean), 3),
        "frac_mu_outside_u_clip_box": round(
            float((np.abs(mix_mean[valid]) > u_clip).any(-1).mean()), 4),
        "width_vs_prior": _stats(width[valid]),
        "frac_wider_than_prior": round(float((width[valid] > 1.0).mean()), 4),
    }

    # --- what drives ESS: geometry (mismatch) or noise? (cf. HANDOFF 07-29) ---
    if ess is not None:
        report["ess_correlations"] = {
            "vs_cov_trace": _corr(ess[valid], tr.sum(1)[valid]),
            "vs_mu_norm": _corr(ess[valid], mu_norm[valid]),
            "vs_roundtrip": _corr(ess[valid], rt[valid]) if rt is not None else None,
        }

    arrays = {"ess": ess, "roundtrip": rt, "sq_norm": sq,
              "mu_norm": mu_norm, "width": width, "valid": valid,
              "obs": data.get("observations")}
    return report, arrays


def plot(report, arrays, out_png, u_clip):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, INK, MUTED, ALERT = "#3d6fb0", "#1f2430", "#6b7280", "#b3563d"
    d_a = report["action_dim"]
    valid = arrays["valid"]
    panels = [p for p in [
        ("ess", "ESS (final EM iterate)", ESS_GATE, "D3 gate 20"),
        ("roundtrip", "round-trip  ||G(s,E(s,a)) - a||", ROUNDTRIP_GATE, "gate 0.1"),
        ("sq_norm", "||u*||^2  (typicality vs chi^2)", None, None),
        ("mu_norm", "||posterior mean||", u_clip, f"u_clip {u_clip:g}"),
        ("width", "posterior width / prior", 1.0, "prior = 1"),
    ] if arrays[p[0]] is not None]

    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.0))
    for ax, (key, title, ref, ref_label) in zip(np.atleast_1d(axes), panels):
        x = np.asarray(arrays[key], np.float64)[valid]
        x = x[np.isfinite(x)]
        hi = np.percentile(x, 99.5)
        ax.hist(x[x <= hi], bins=80, color=BLUE, edgecolor="none")
        if key == "sq_norm":
            try:
                from scipy.stats import chi2
                g = np.linspace(0, hi, 200)
                dens = chi2.pdf(g, d_a)
                counts, edges = np.histogram(x[x <= hi], bins=80)
                ax.plot(g, dens * counts.max() / max(dens.max(), 1e-12),
                        color=INK, lw=1.5, label=f"chi2_{d_a} (scaled)")
                ax.legend(frameon=False, fontsize=7)
            except ImportError:
                pass
        if ref is not None:
            ax.axvline(ref, color=ALERT, lw=1.2, ls="--")
            ax.text(ref, ax.get_ylim()[1] * 0.95, f" {ref_label}", color=ALERT,
                    fontsize=7, va="top")
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
    src = report.get("meta", {})
    fig.suptitle(
        f"preimages: n={report['n']:,}  alpha={src.get('inversion', {}).get('alpha', '?')}"
        f"  N={src.get('inversion', {}).get('num_samples', '?')}"
        f"  invalid={report['invalid']['frac']:.2%}",
        fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_png, dpi=150)
    print(f"figure -> {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz")
    ap.add_argument("--u-clip", type=float, default=U_CLIP_DEFAULT)
    ap.add_argument("--out", default=None, help="JSON path; default <npz>.analysis.json")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    report, arrays = analyze(args.npz, u_clip=args.u_clip)
    out = args.out or args.npz + ".analysis.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"report -> {out}")
    if not args.no_plot:
        plot(report, arrays, (args.out or args.npz) + ".analysis.png", args.u_clip)


if __name__ == "__main__":
    main()
