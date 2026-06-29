"""scripts/probes/representation_profile_aggregate.py — 7-seed aggregate.

Reads <root>/s*_final/{value_landscape,value_steps,z_decoding,
b_resolution,coverage}.parquet, tags `seed`, writes mean±std tables,
plots, and a data-filled story.md.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SEED_RE = re.compile(r"s(\d+)_final")

from evals._profile_core import (  # noqa: E402
    _synthesis, _verdict_t1, _verdict_t2, _verdict_t3, _verdict_t4,
)


def _load(root: Path, name: str) -> pd.DataFrame:
    frames = []
    for p in sorted(root.glob(f"s*_final/{name}.parquet")):
        m = _SEED_RE.search(p.parent.name)
        if not m:
            continue
        df = pd.read_parquet(p)
        df["seed"] = int(m.group(1))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No s*_final/{name}.parquet under {root}. Run "
            f"scripts/probes/representation_profile.py for each seed first.")
    return pd.concat(frames, ignore_index=True)


def _ms(series: pd.Series) -> str:
    return f"{series.mean():.3g} ± {series.std(ddof=0):.2g}"


def aggregate(root, out) -> None:
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)

    vl = _load(root, "value_landscape")
    vs = _load(root, "value_steps")
    zd = _load(root, "z_decoding")
    br = _load(root, "b_resolution")
    cv = _load(root, "coverage")

    # T1 value gradient: per task, slope by outcome (mean over seeds)
    t1 = (vl.groupby(["task", "outcome"])["rho_V_negd"]
          .agg(["mean", "std"]).reset_index())
    t1.to_parquet(out / "T1_value_gradient.parquet")
    t2 = (zd.groupby("task")[["relabel_pos_frac", "topk_mean_d",
                              "topk_pct_at_goal"]]
          .agg(["mean", "std"]))
    t2.to_parquet(out / "T2_sparsity.parquet")
    t3 = (br.groupby("task")[["r2", "placed_vs_near_acc"]]
          .agg(["mean", "std"]))
    t3.to_parquet(out / "T3_b_resolution.parquet")
    t4 = (cv.groupby(["region", "outcome"])["nn_dist"]
          .agg(["mean", "std"]).reset_index())
    t4.to_parquet(out / "T4_coverage.parquet")

    # value_vs_dist.png — V binned by d, success vs transport_fail
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0.0, max(0.05, float(vs["d"].quantile(0.95))), 9)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for oc, color in (("success", "#55a868"),
                      ("transport_fail", "#c44e52")):
        sub = vs[vs.outcome == oc]
        if sub.empty:
            continue
        idx = np.clip(np.digitize(sub["d"], bins) - 1, 0, len(centers) - 1)
        g = pd.DataFrame({"b": idx, "V": sub["V"].values}).groupby("b")["V"]
        ax.plot(centers[g.mean().index], g.mean().values, marker="o",
                label=oc, color=color)
    ax.set_xlabel("cube → goal distance")
    ax.set_ylabel("V(s)")
    ax.set_title("Value vs goal distance (S0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "value_vs_dist.png", dpi=120)
    plt.close(fig)

    for name, frame, col in (
        ("sparsity_bars", zd, "topk_pct_at_goal"),
        ("b_resolution_bars", br, "r2")):
        fig, ax = plt.subplots(figsize=(6, 4))
        gg = frame.groupby("task")[col].mean()
        ax.bar([str(t) for t in gg.index], gg.values)
        ax.set_title(name)
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=20, labelsize=7)
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    cc = cv.groupby(["region", "outcome"])["nn_dist"].mean().unstack()
    cc.plot(kind="bar", ax=ax)
    ax.set_title("coverage_curves")
    ax.set_ylabel("nn_dist")
    fig.tight_layout()
    fig.savefig(out / "coverage_curves.png", dpi=120)
    plt.close(fig)

    succ = vl[vl.outcome == "success"]["rho_V_negd"]
    fail = vl[vl.outcome == "transport_fail"]["rho_V_negd"]
    rho_s, rho_f = float(succ.mean()), float(fail.mean())
    frac = float(zd["relabel_pos_frac"].mean())
    tkg = float(zd["topk_pct_at_goal"].mean())
    r2 = float(br["r2"].mean())
    acc = float(br["placed_vs_near_acc"].mean())
    cvt = cv[cv.region == "transport"]
    fail_nn = float(cvt[cvt.outcome == "transport_fail"]["nn_dist"].mean())
    succ_nn = float(cvt[cvt.outcome == "success"]["nn_dist"].mean())

    verdicts = {
        "T1": _verdict_t1(rho_s, rho_f),
        "T2": _verdict_t2(frac, tkg),
        "T3": _verdict_t3(r2, acc),
        "T4": _verdict_t4(fail_nn, succ_nn),
    }
    lines = [
        "# FB representation-failure — cross-seed story", "",
        f"Seeds: {', '.join('s'+str(s) for s in sorted(vl['seed'].unique()))}",
        "",
        "## Sparsity (T2)",
        f"- offline goal-state fraction: {_ms(zd['relabel_pos_frac'])}",
        f"- top-k B·z mean cube→goal dist: {_ms(zd['topk_mean_d'])}",
        f"- top-k % actually at goal: {_ms(zd['topk_pct_at_goal'])}",
        "",
        "## B resolution (T3)",
        f"- ridge R²(B→d): {_ms(br['r2'])}",
        f"- placed-vs-near-miss acc: {_ms(br['placed_vs_near_acc'])}",
        "",
        "## Value gradient (T1)",
        f"- Spearman ρ(V vs −d) success: {_ms(succ)}",
        f"- Spearman ρ(V vs −d) transport-fail: {_ms(fail)}",
        "",
        "## Coverage (T4)",
        f"- transport-region nn_dist (transport_fail): "
        f"{_ms(cv[(cv.region=='transport') & (cv.outcome=='transport_fail')]['nn_dist'])}",
        "",
        "## Readout",
        "_Heuristic verdicts (thresholds in "
        "representation_profile_aggregate.py):_",
        f"- T1 value gradient: rho_success={rho_s:.3g}, "
        f"rho_fail={rho_f:.3g} -> {verdicts['T1'][0]} ({verdicts['T1'][1]})",
        f"- T2 sparsity: relabel_pos_frac={frac:.3g}, "
        f"topk_pct_at_goal={tkg:.3g} -> {verdicts['T2'][0]} "
        f"({verdicts['T2'][1]})",
        f"- T3 B-resolution: r2={r2:.3g}, placed_vs_near_acc={acc:.3g} "
        f"-> {verdicts['T3'][0]} ({verdicts['T3'][1]})",
        f"- T4 coverage: fail={fail_nn:.3g}, success={succ_nn:.3g} "
        f"(transport) -> {verdicts['T4'][0]} ({verdicts['T4'][1]})",
        _synthesis(verdicts),
    ]
    (out / "story.md").write_text("\n".join(lines))
    print(f"[representation_profile_aggregate] "
          f"{vl['seed'].nunique()} seeds -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",
                    default=str(REPO_ROOT / "analysis" / "probes" / "representation_profile"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    aggregate(root, Path(args.out) if args.out else root / "aggregate")


if __name__ == "__main__":
    main()
