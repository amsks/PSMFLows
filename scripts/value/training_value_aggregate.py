"""scripts/value/training_value_aggregate.py — Spearman table + value-vs-distance
curves + phase-composition bar + value-over-cube-xy scenes for the
training-data value analysis. Reads fb_values/gciql_values/phase_composition
from --root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.training_value import (phase_spearman_table, outcome_spearman_table,
                                   value_discrimination)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REGIONS = ["reach", "grasp", "transport"]
# Display names: per-state regimes aligned with the paper (the stored labels
# reach/grasp/transport predate the {approach, pickup, carry} regime naming).
REGION_DISPLAY = {"reach": "approach", "grasp": "pickup", "transport": "carry"}
OUTCOMES = ["success_bound", "fail_bound"]
HORIZON_DEFAULT = 200    # label provenance (see profile --horizon)
THRESH_DEFAULT = 0.04    # label provenance (see profile --thresh)


def _load_pooled(root: Path, name: str):
    """Read `name` from per-pair subdirs p*/ (multi-seed, tagged with `pair`)
    if present, else from `root` directly (single run)."""
    pair_dirs = sorted(d for d in root.glob("p*") if d.is_dir())
    if pair_dirs:
        frames = []
        for pd_ in pair_dirs:
            f = pd_ / name
            if f.exists():
                df = pd.read_parquet(f)
                df["pair"] = pd_.name
                frames.append(df)
        return (pd.concat(frames, ignore_index=True) if frames else None), True
    f = root / name
    return (pd.read_parquet(f) if f.exists() else None), False


def _methods(root: Path):
    """((df, value_col, label) list, group_cols). Pools per-pair subdirs."""
    fb, multi = _load_pooled(root, "fb_values.parquet")
    out = [(fb, "V_policy", "FB policy"), (fb, "V_data", "FB data")]
    gq, _ = _load_pooled(root, "gciql_values.parquet")
    if gq is not None:
        out.append((gq, "V", "GCIQL"))
    cr, _ = _load_pooled(root, "crl_values.parquet")
    if cr is not None:
        out.append((cr, "V", "CRL"))
    rl, _ = _load_pooled(root, "rldp_values.parquet")
    if rl is not None:
        out.append((rl, "V", "RLDP"))
    group_cols = ("pair", "task") if multi else ("task",)
    return out, group_cols


def aggregate(root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    methods, group_cols = _methods(root)
    n_seeds = (methods[0][0]["pair"].nunique()
               if "pair" in methods[0][0].columns else 1)

    # 1. Spearman table (region x method, mean over group_cols).
    rows = []
    for df, vc, label in methods:
        tab = phase_spearman_table(df, value_col=vc, group_cols=group_cols)
        for _, r in tab.iterrows():
            rows.append({"method": label, "region": r["region"],
                         "rho_mean": r["rho_mean"], "rho_std": r["rho_std"]})
    sp = pd.DataFrame(rows)
    sp.to_parquet(out / "spearman_table.parquet")
    md = ["# Training-state value gradient: Spearman rho(V, -d) by phase",
          f"_n_seeds={n_seeds}; mean over {'seed x task' if n_seeds > 1 else 'task'}_",
          "",
          "| method | " + " | ".join(REGIONS) + " |",
          "| :--- | " + " | ".join([":---"] * len(REGIONS)) + " |"]
    for label in [m[2] for m in methods]:
        cells = [label]
        for rg in REGIONS:
            s = sp[(sp.method == label) & (sp.region == rg)]
            cells.append(f"{s['rho_mean'].iloc[0]:.3f}" if len(s) else "—")
        md.append("| " + " | ".join(cells) + " |")
    (out / "spearman_table.md").write_text("\n".join(md) + "\n")

    # 2. value-vs-distance curves (faceted by phase, line per method, z-scored).
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(5 * len(REGIONS), 4),
                             squeeze=False)
    for ax, rg in zip(axes[0], REGIONS):
        for df, vc, label in methods:
            sub = df[df.region == rg]
            if sub.empty:
                continue
            v = sub[vc].to_numpy(float)
            vz = (v - v.mean()) / (v.std() + 1e-9)
            edges = np.unique(np.quantile(sub["d"], np.linspace(0, 1, 10)))
            if len(edges) < 3:
                continue
            ctr = 0.5 * (edges[:-1] + edges[1:])
            idx = np.clip(np.digitize(sub["d"], edges) - 1, 0, len(ctr) - 1)
            mv = pd.DataFrame({"b": idx, "vz": vz}).groupby("b")["vz"].mean()
            ax.plot(ctr[mv.index], mv.values, marker="o", label=label)
        ax.set_title(REGION_DISPLAY.get(rg, rg))
        ax.set_xlabel("cube -> goal distance")
        ax.set_ylabel("z-scored value")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Training states: value vs cube-to-goal distance, by phase")
    fig.tight_layout()
    fig.savefig(out / "value_vs_dist.png", dpi=130)
    plt.close(fig)

    # 3. dataset phase composition bar (averaged across pairs if multi-seed).
    comp_raw, _ = _load_pooled(root, "phase_composition.parquet")
    comp = (comp_raw.groupby("region")["fraction"].mean().reset_index()
            .set_index("region").reindex(REGIONS).fillna(0.0))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(REGIONS, comp["fraction"].values)
    ax.set_ylabel("fraction of training transitions")
    ax.set_title("Dataset phase composition")
    fig.tight_layout()
    fig.savefig(out / "phase_composition.png", dpi=130)
    plt.close(fig)

    # 4. value-over-cube-xy scenes (phase x method): binned mean value.
    for df, vc, label in methods:
        fig, axes = plt.subplots(1, len(REGIONS),
                                 figsize=(5 * len(REGIONS), 4), squeeze=False)
        for ax, rg in zip(axes[0], REGIONS):
            sub = df[df.region == rg]
            ax.set_title(f"{label} — {rg} (n={len(sub)})")
            if len(sub) >= 20:
                h = ax.hexbin(sub["cube_x"], sub["cube_y"], C=sub[vc],
                              gridsize=20, reduce_C_function=np.mean,
                              cmap="Reds")
                fig.colorbar(h, ax=ax, shrink=0.8)
            ax.set_xlabel("cube x")
            ax.set_ylabel("cube y")
        fig.tight_layout()
        fname = f"scene_{label.replace(' ', '_').lower()}.png"
        fig.savefig(out / fname, dpi=130)
        plt.close(fig)

    # 5-8. Outcome-split outputs (skip cleanly on legacy dirs w/o 'outcome').
    if not all("outcome" in df.columns for df, _, _ in methods):
        print("[training_value_aggregate] no 'outcome' column — skipping "
              "outcome-split outputs (re-run profile passes).")
    else:
        # 5. Spearman rho(V,-d) by phase x outcome.
        rows = []
        for df, vc, label in methods:
            tab = outcome_spearman_table(df, value_col=vc,
                                         group_cols=group_cols)
            for _, r in tab.iterrows():
                rows.append({"method": label, "region": r["region"],
                             "outcome": r["outcome"],
                             "rho_mean": r["rho_mean"],
                             "rho_std": r["rho_std"]})
        spo = pd.DataFrame(rows)
        spo.to_parquet(out / "spearman_outcome_table.parquet")
        mdo = ["# Training-state value gradient by phase x outcome: "
               "Spearman rho(V, -d)",
               f"_n_seeds={n_seeds}; success-bound = cube within "
               f"{THRESH_DEFAULT} m of goal within {HORIZON_DEFAULT} steps. "
               "Pathology replicates if rho is NOT higher for success_bound._",
               "",
               "| method | outcome | " + " | ".join(REGIONS) + " |",
               "| :--- | :--- | " + " | ".join([":---"] * len(REGIONS)) + " |"]
        for label in [m[2] for m in methods]:
            for oc in OUTCOMES:
                cells = [label, oc]
                for rg in REGIONS:
                    s = spo[(spo.method == label) & (spo.region == rg)
                            & (spo.outcome == oc)]
                    cells.append(f"{s['rho_mean'].iloc[0]:.3f}"
                                 if len(s) else "—")
                mdo.append("| " + " | ".join(cells) + " |")
        (out / "spearman_outcome_table.md").write_text("\n".join(mdo) + "\n")

        # 6. Outcome-discrimination: AUC + mean dV.
        rows = []
        for df, vc, label in methods:
            disc = value_discrimination(df, value_col=vc,
                                        group_cols=group_cols)
            for _, r in disc.iterrows():
                rows.append({"method": label, "region": r["region"],
                             "auc": r["auc"], "mean_dV": r["mean_dV"],
                             "n_success": r["n_success"],
                             "n_fail": r["n_fail"]})
        disc_df = pd.DataFrame(rows)
        disc_df.to_parquet(out / "value_discrimination.parquet")
        mdd = ["# Value outcome-discrimination on training states",
               "_AUC of value separating success-bound vs fail-bound "
               "(0.5 = none); (dV) = z-scored mean value gap_", "",
               "| method | "
               + " | ".join(f"{rg} AUC (dV)" for rg in REGIONS) + " |",
               "| :--- | " + " | ".join([":---"] * len(REGIONS)) + " |"]
        for label in [m[2] for m in methods]:
            cells = [label]
            for rg in REGIONS:
                s = disc_df[(disc_df.method == label) & (disc_df.region == rg)]
                if len(s) and np.isfinite(s["auc"].iloc[0]):
                    cells.append(f"{s['auc'].iloc[0]:.2f} "
                                 f"({s['mean_dV'].iloc[0]:+.2f})")
                else:
                    cells.append("—")
            mdd.append("| " + " | ".join(cells) + " |")
        (out / "value_discrimination.md").write_text("\n".join(mdd) + "\n")

        # 7. value-vs-distance by outcome (rows=method, cols=phase).
        n_m = len(methods)
        fig, axes = plt.subplots(n_m, len(REGIONS),
                                 figsize=(5 * len(REGIONS), 4 * n_m),
                                 squeeze=False)
        for mi, (df, vc, label) in enumerate(methods):
            for ax, rg in zip(axes[mi], REGIONS):
                for oc, color in (("success_bound", "tab:green"),
                                  ("fail_bound", "tab:red")):
                    sub = df[(df.region == rg) & (df.outcome == oc)]
                    if len(sub) < 20:
                        continue
                    v = sub[vc].to_numpy(float)
                    vz = (v - v.mean()) / (v.std() + 1e-9)
                    edges = np.unique(np.quantile(sub["d"],
                                                  np.linspace(0, 1, 8)))
                    if len(edges) < 3:
                        continue
                    ctr = 0.5 * (edges[:-1] + edges[1:])
                    idx = np.clip(np.digitize(sub["d"], edges) - 1,
                                  0, len(ctr) - 1)
                    mv = pd.DataFrame({"b": idx, "vz": vz}).groupby(
                        "b")["vz"].mean()
                    ax.plot(ctr[mv.index], mv.values, marker="o",
                            color=color, label=oc)
                ax.set_title(f"{label} — {rg}")
                ax.set_xlabel("cube -> goal distance")
                ax.set_ylabel("z-scored value")
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
        fig.suptitle("Training states: value vs distance, by phase and outcome")
        fig.tight_layout()
        fig.savefig(out / "value_vs_dist_by_outcome.png", dpi=130)
        plt.close(fig)

        # 8. Outcome composition (success-bound fraction per phase) — sanity.
        comp_df = methods[0][0].assign(
            _succ=(methods[0][0]["outcome"] == "success_bound").astype(float))
        by_region = (comp_df.groupby("region")["_succ"].agg(["mean", "size"])
                     .reindex(REGIONS))
        by_region.reset_index().rename(
            columns={"mean": "frac_success_bound", "size": "n"}
        ).to_parquet(out / "outcome_composition.parquet")
        mdc = ["# Success-bound fraction on training states (per phase)",
               "_fraction of (state x task) rows that are success_bound_", "",
               "| region | frac success_bound | n |", "| :--- | :--- | :--- |"]
        for rg in REGIONS:
            if rg in by_region.index and np.isfinite(by_region.loc[rg, "mean"]):
                mdc.append(f"| {rg} | {by_region.loc[rg, 'mean']:.3f} | "
                           f"{int(by_region.loc[rg, 'size'])} |")
            else:
                mdc.append(f"| {rg} | — | 0 |")
        (out / "outcome_composition.md").write_text("\n".join(mdc) + "\n")

        # 9. Spatial value maps over cube-xy, split by outcome (per method).
        #    Value z-scored per (group) so seeds/tasks pool; shared color
        #    scale per phase row so success vs fail are comparable.
        for df, vc, label in methods:
            d = df.copy()
            d["_vz"] = d.groupby(list(group_cols))[vc].transform(
                lambda s: (s - s.mean()) / (s.std() + 1e-9))
            fig, axes = plt.subplots(
                len(REGIONS), len(OUTCOMES),
                figsize=(5 * len(OUTCOMES), 4 * len(REGIONS)),
                squeeze=False, constrained_layout=True)
            for ri, rg in enumerate(REGIONS):
                row = d[d.region == rg]
                vmin = (float(np.nanpercentile(row["_vz"], 2))
                        if len(row) else -1.0)
                vmax = (float(np.nanpercentile(row["_vz"], 98))
                        if len(row) else 1.0)
                last_h = None
                for ci, oc in enumerate(OUTCOMES):
                    ax = axes[ri][ci]
                    sub = row[row.outcome == oc]
                    ax.set_title(f"{label} — {rg} — {oc} (n={len(sub)})")
                    if len(sub) >= 20:
                        last_h = ax.hexbin(sub["cube_x"], sub["cube_y"],
                                           C=sub["_vz"], gridsize=20,
                                           reduce_C_function=np.mean,
                                           cmap="viridis", vmin=vmin, vmax=vmax)
                    ax.set_xlabel("cube x")
                    ax.set_ylabel("cube y")
                if last_h is not None:
                    fig.colorbar(last_h, ax=list(axes[ri]), shrink=0.8,
                                 label="z-scored value")
            fig.suptitle(f"{label}: training-state value over cube-xy, "
                         "by phase x outcome")
            fname = f"scene_outcome_{label.replace(' ', '_').lower()}.png"
            fig.savefig(out / fname, dpi=130)
            plt.close(fig)

        print("\n".join(mdo))
        print("\n".join(mdd))

    print(f"[training_value_aggregate] -> {out}")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="analysis/value/training_value")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    aggregate(root, Path(args.out) if args.out else root / "aggregate")


if __name__ == "__main__":
    main()
