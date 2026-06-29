#!/usr/bin/env python
"""scripts/figures/plot_repsep.py — representation-aliasing figure (rlbrew P4).

Reads the per-method grasp-state representations in analysis/value/repsep/*.parquet
(columns {split, task, success, rep_0..rep_k}) and renders, as the headline P4
visual, that the learned representations do NOT separate soon-to-succeed from
soon-to-fail in-hand states.

For each method we fit a 2D PCA on its pooled (train+eval) grasps and a linear
probe (logistic regression, stratified 5-fold, ROC-AUC) separately on train and
eval grasps. The figure is a 2x4 grid: rows {Train, Eval} x cols {FB, GCIQL,
CRL, RLDP}; each point is an in-hand state coloured green=success-bound /
vermillion=fail-bound, each panel annotated with its linear-probe AUC. AUC ~0.5
= no linear separation = aliasing; the train->eval AUC drop = overgeneralisation.

Outputs PAPER/rlbrew/figures/fig_repsep.png and analysis/value/repsep/auc_table.md.

Run under .venv (sklearn + seaborn).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent.parent
REPSEP = REPO / "analysis" / "value" / "repsep"
OUT = REPO / "PAPER" / "rlbrew" / "figures"

sns.set_palette("colorblind")
_CB = sns.color_palette("colorblind").as_hex()
SUCC, FAIL = _CB[2], _CB[3]          # green / vermillion

METHODS = ["fb", "gciql", "crl", "rldp"]
LABELS = {"fb": "FB", "gciql": "GCIQL", "crl": "CRL", "rldp": "RLDP"}
MAX_PTS = 600                        # per class per panel (visual clarity)
RNG = np.random.default_rng(0)


def _load(method):
    df = pd.read_parquet(REPSEP / f"{method}.parquet")
    cols = sorted((c for c in df.columns if c.startswith("rep_")),
                  key=lambda c: int(c.split("_")[1]))
    X = df[cols].to_numpy(np.float32)
    y = df["success"].to_numpy(bool)
    split = df["split"].to_numpy()
    episode = df["episode"].to_numpy(int)
    task = df["task"].to_numpy(int)
    return X, y, split, episode, task


def _auc_per_task(X, y, task):
    """Mean per-task AUC: success-bound is task-dependent, so probe each task
    separately (fixed goal) and average -- avoids the contradictory-label
    pooling across tasks."""
    aucs = [_auc(X[task == t], y[task == t]) for t in np.unique(task)]
    return float(np.nanmean(aucs))


def _auc(X, y, groups=None):
    """ROC-AUC of a standardised logistic probe, 5-fold.

    With `groups` (eval: episode ids) use StratifiedGroupKFold so no rollout
    spans train/test folds -- otherwise per-step labels leak episode identity
    and the AUC is meaningless. Train grasps are independent -> StratifiedKFold.
    """
    if len(np.unique(y)) < 2 or min(np.bincount(y.astype(int))) < 5:
        return np.nan
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0))
    if groups is not None:
        n_groups = len(np.unique(groups))
        n_splits = min(5, n_groups)
        if n_splits < 2:
            return np.nan
        cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=0)
        return float(cross_val_score(clf, X, y.astype(int), groups=groups,
                                     cv=cv, scoring="roc_auc").mean())
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    return float(cross_val_score(clf, X, y.astype(int), cv=cv,
                                 scoring="roc_auc").mean())


def _subsample(idx, y):
    """Balanced subsample of indices for a scatter panel."""
    pos = idx[y[idx]]
    neg = idx[~y[idx]]
    pos = RNG.choice(pos, min(len(pos), MAX_PTS), replace=False)
    neg = RNG.choice(neg, min(len(neg), MAX_PTS), replace=False)
    return pos, neg


def _panel_pts(ax, Z2, y2, auc, title=None, ylab=None):
    """Scatter already-embedded 2D points Z2 with labels y2."""
    pos = np.where(y2)[0]
    neg = np.where(~y2)[0]
    ax.scatter(Z2[neg, 0], Z2[neg, 1], s=6, c=FAIL, alpha=0.35,
               linewidths=0, label="Fail-bound")
    ax.scatter(Z2[pos, 0], Z2[pos, 1], s=6, c=SUCC, alpha=0.45,
               linewidths=0, label="Success-bound")
    # class centroids make the overlap legible
    for sub, col in ((pos, SUCC), (neg, FAIL)):
        if len(sub):
            ax.scatter(Z2[sub, 0].mean(), Z2[sub, 1].mean(), s=160, marker="X",
                       c=col, edgecolors="black", linewidths=1.2, zorder=5)
    txt = "AUC n/a" if np.isnan(auc) else f"AUC {auc:.2f}"
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.85))
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    if ylab:
        ax.set_ylabel(ylab, fontsize=12, fontweight="bold")


def _embed_subset(X, projection, pca_full=None, sel=None):
    """Return 2D coords for the selected rows under `projection`.

    PCA: index the precomputed pooled fit (panels share axes). t-SNE: fit on the
    standardised subset directly (per-panel; t-SNE has no shared transform)."""
    if projection == "pca":
        return pca_full[sel]
    Xs = StandardScaler().fit_transform(X[sel])
    perp = float(min(30, max(5, (len(sel) - 1) // 3)))
    return TSNE(n_components=2, perplexity=perp, init="pca",
                random_state=0).fit_transform(Xs)


def _render_grid(out, data, pca, auc_map, projection, title):
    """2x4 grid (rows train/eval x cols methods) under `projection`."""
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.4))
    for j, m in enumerate(METHODS):
        X, y, split, _ = data[m]
        for r, (sp, ylab) in enumerate((("train", "Train grasps"),
                                        ("eval", "Eval grasps"))):
            idx = np.where(split == sp)[0]
            pos, neg = _subsample(idx, y)
            sel = np.concatenate([pos, neg])
            Z2 = _embed_subset(X, projection, pca_full=pca[m], sel=sel)
            _panel_pts(axes[r, j], Z2, y[sel], auc_map[m][r],
                       title=LABELS[m] if r == 0 else None,
                       ylab=ylab if j == 0 else None)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=7, c=SUCC,
                          label="Success-bound"),
               plt.Line2D([], [], marker="o", ls="", ms=7, c=FAIL,
                          label="Fail-bound")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data, pca, auc_map, auc_rows = {}, {}, {}, []
    for m in METHODS:
        X, y, split, episode, task = _load(m)
        data[m] = (X, y, split, episode)
        # per-method PCA fit on pooled grasps so train/eval share axes
        Xs = StandardScaler().fit_transform(X)
        pca[m] = PCA(n_components=2, random_state=0).fit_transform(Xs)
        tr, ev = split == "train", split == "eval"
        auc_tr = _auc_per_task(X[tr], y[tr], task[tr])  # per-task (robust)
        auc_ev = _auc(X[ev], y[ev], groups=episode[ev])  # high-variance
        auc_map[m] = (auc_tr, auc_ev)
        auc_rows.append((LABELS[m], auc_tr, auc_ev))
        print(f"[repsep] {LABELS[m]}: train AUC={auc_tr:.3f} "
              f"eval AUC={auc_ev:.3f} (n_tr={tr.sum()}, n_ev={ev.sum()})")

    _render_grid(OUT / "fig_repsep.png", data, pca, auc_map, "pca",
                 "In-hand grasp representations: success-bound vs fail-bound "
                 "separability (per-method PCA; linear-probe AUC)")
    _render_grid(OUT / "fig_repsep_tsne.png", data, pca, auc_map, "tsne",
                 "In-hand grasp representations: success-bound vs fail-bound "
                 "separability (per-method t-SNE; linear-probe AUC)")

    # AUC table for the appendix
    lines = ["| Method | Train AUC | Eval AUC | Train$\\to$Eval drop |",
             "|---|---|---|---|"]
    for name, a_tr, a_ev in auc_rows:
        drop = "n/a" if (np.isnan(a_tr) or np.isnan(a_ev)) else f"{a_tr - a_ev:+.2f}"
        lines.append(f"| {name} | {a_tr:.2f} | {a_ev:.2f} | {drop} |")
    tbl = REPSEP / "auc_table.md"
    tbl.write_text("\n".join(lines) + "\n")
    print(f"[plot] wrote {tbl}\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
