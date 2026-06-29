"""scripts/dev/compare_td_jepa.py — Side-by-side comparison of our verify-ortho-fix run
vs td_jepa's run on the same data + hyperparameters.

Pulls both runs from wandb (project: factored-fb), aligns metric trajectories
by step, and prints a delta table.

Both codebases use identical metric key names under the "train/" prefix.

Must be run with td_jepa's uv environment (needs pandas + wandb from td_jepa):

    uv run --directory /home/mclovin/git/Austin/td_jepa \\
        python /home/mclovin/git/Austin/Factored-FB/scripts/dev/compare_td_jepa.py

    uv run --directory /home/mclovin/git/Austin/td_jepa \\
        python /home/mclovin/git/Austin/Factored-FB/scripts/dev/compare_td_jepa.py \\
        --ours-group verify-ortho-fix --tdjepa-group td-jepa-50k
"""

from __future__ import annotations

import argparse
from typing import Optional

import pandas as pd
import wandb


# Keys: (ours_in_wandb, tdjepa_in_wandb) — identical names since standardization
METRIC_PAIRS = {
    "M1":           ("train/M1",              "train/M1"),
    "q":            ("train/q",               "train/q"),
    "fb_loss":      ("train/fb_loss",         "train/fb_loss"),
    "fb_diag":      ("train/fb_diag",         "train/fb_diag"),
    "fb_offdiag":   ("train/fb_offdiag",      "train/fb_offdiag"),
    "orth_loss":    ("train/orth_loss",        "train/orth_loss"),
    "orth_offdiag": ("train/orth_loss_offdiag", "train/orth_loss_offdiag"),
    "actor_loss":   ("train/actor_loss",      "train/actor_loss"),
    "bc_flow":      ("train/bc_flow_loss",    "train/bc_flow_loss"),
    "B_norm":       ("train/B_norm",          "train/B_norm"),
    "z_norm":       ("train/z_norm",          "train/z_norm"),
}


def latest_run_in_group(api: wandb.Api, project: str, group: str) -> Optional[wandb.apis.public.Run]:
    runs = list(api.runs(project, filters={"group": group}, order="-created_at"))
    return runs[0] if runs else None


def pull_history(run: wandb.apis.public.Run) -> pd.DataFrame:
    hist = run.history(samples=10_000, pandas=True)
    return hist.sort_values("_step").reset_index(drop=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default="amsks/factored-fb")
    ap.add_argument("--ours-group", default="verify-ortho-fix")
    ap.add_argument("--tdjepa-group", default="td-jepa-50k")
    return ap.parse_args()


def main():
    args = parse_args()
    api = wandb.Api()

    ours = latest_run_in_group(api, args.project, args.ours_group)
    tdj = latest_run_in_group(api, args.project, args.tdjepa_group)
    if ours is None:
        raise SystemExit(f"No runs in group {args.ours_group!r}")
    if tdj is None:
        raise SystemExit(f"No runs in group {args.tdjepa_group!r}")

    print(f"ours:    {ours.name}  id={ours.id}  state={ours.state}")
    print(f"td_jepa: {tdj.name}  id={tdj.id}  state={tdj.state}")
    print()

    ours_df = pull_history(ours)
    tdj_df = pull_history(tdj)

    # Pick reference steps: every 5000 up to 50000 (or whichever is shorter)
    ref_steps = [s for s in range(5_000, 50_001, 5_000)
                 if s <= ours_df["_step"].max() and s <= tdj_df["_step"].max()]

    def value_at(df: pd.DataFrame, key: str, step: int) -> Optional[float]:
        if key not in df.columns:
            return None
        sub = df.loc[df["_step"] <= step, ["_step", key]].dropna()
        if sub.empty:
            return None
        return float(sub.iloc[-1][key])

    print(f"{'metric':14s} {'step':>7s}  {'ours':>10s}  {'td_jepa':>10s}  {'Δ':>10s}  {'Δ/td_jepa':>10s}")
    print("-" * 76)
    for label, (ours_key, tdj_key) in METRIC_PAIRS.items():
        for step in ref_steps:
            o = value_at(ours_df, ours_key, step)
            t = value_at(tdj_df, tdj_key, step)
            if o is None or t is None:
                continue
            delta = o - t
            rel = (delta / t) if abs(t) > 1e-6 else float("nan")
            print(f"{label:14s} {step:>7d}  {o:>10.3f}  {t:>10.3f}  {delta:>+10.3f}  {rel:>+10.2%}")
        print()


if __name__ == "__main__":
    main()
