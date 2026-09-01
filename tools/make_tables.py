"""Assemble the two results tables from the 500-episode eval JSONs.

Hand-typed tables go stale the moment a queued seed lands, and this project now has three
ACTING MODES per checkpoint (deployed / decode-only / lambda-rank) whose numbers differ by
more than the effects being reported. So the mapping from "table cell" to "which eval JSONs"
is written out explicitly below, and everything else is derived.

Aggregation follows the project convention: more than one seed -> mean +/- 95% CI across
seeds (t interval); a single seed -> that run's Wilson interval, marked. A cell whose files
are missing prints as "--" and is listed under `missing` in the JSON, so a half-finished
table can never read as a complete one.

Writes:
  PAPER/ICLR/tables/table_headline.tex, table_fraction.tex   (booktabs, \\input-able)
  docs/tables/results.md                                     (same numbers, markdown)
  <logs>/table_dataset_fraction.json                         (machine-readable, incl. gaps)

Run: .venv/bin/python tools/make_tables.py [--logs DIR]
"""
import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOGS = "/data-local/amsks/PSMFLows/logs"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- cell -> eval files. Globs are resolved; every match is one seed. -----------------
# Kept explicit: filename conventions have drifted (lambdarank/decodeonly/1M suffixes), and
# a glob like eval500_hybrid_*.json would silently mix three acting modes into one number.
HEADLINE = [
    ("FQL (per-task reference, raw actions)", "per-task", ["eval500_fql_cube_sd?.json"]),
    ("Latent RL, per-task, eps=0.05 @peak", "per-task", ["eval500_l1stab_res0.05_*_peak*.json"]),
    ("Latent RL, per-task, eps=0 (pure decode)", "per-task", ["eval500_w4_res0.0_sd?_final.json"]),
    ("FB (zero-shot, raw actions)", "zero-shot", ["eval500_fb_cube_sd?.json"]),
    ("PSMFlow (zero-shot, latent -> frozen decode)", "zero-shot", ["eval500_latentpsm_cube_sd?.json"]),
    ("PSMFlow, HP-matched to FB", "zero-shot", ["eval500_hpmatch_sd00?.json"]),
    ("Hybrid (action critic + residual), deployed", "zero-shot",
     ["eval500_hybrid_sd?_final.json", "eval500_hybrid_1M_sd?.json"]),
    ("Hybrid, decode-only control", "zero-shot",
     ["eval500_hybrid_decodeonly_sd00?.json", "eval500_hybrid_1M_decodeonly_sd00?.json"]),
    ("Hybrid, lambda-rank (K=32, no residual)", "zero-shot",
     ["eval500_hybrid_lambdarank_sd00?.json"]),
    ("Hybrid + FB graft, deployed", "zero-shot", ["eval500_fbgraft_sd?_deployed.json"]),
    ("Hybrid + FB graft, decode-only control", "zero-shot",
     ["eval500_fbgraft_sd?_decodeonly.json"]),
    ("Behavior-cloning control (per-step prior)", "control", ["eval500_bcflow_cube.json"]),
    # E2 (08-31): same 5 checkpoints re-evaluated post-P0.2 seeding, one-step vs ODE-100.
    ("PSMFlow re-eval, actor, one-step decode", "zero-shot", ["e2_onestep_actor_sd?.json"]),
    ("PSMFlow re-eval, actor, exact ODE-100 decode", "zero-shot", ["e2_ode_actor_sd?.json"]),
    ("PSMFlow re-eval, gpi, one-step decode", "zero-shot", ["e2_onestep_gpi_sd?.json"]),
    ("PSMFlow re-eval, gpi, exact ODE-100 decode", "zero-shot", ["e2_ode_gpi_sd?.json"]),
    # E3 (08-31): paper-faithful arms; epoch recorded per-file in table_dataset_fraction.json.
    # `sd?` only, NOT `sd?*`: the mid-training `*_ep250000.json` evals of the runs the
    # 08-31 disk-full killed must not be pooled with the 500k results that replaced them.
    ("Paper-faithful Arm A (u'~p0 bootstrap)", "zero-shot", ["eval500_paperfaith_armA_sd?.json"]),
    ("Paper-faithful Arm B (psi(s,u,u'), no actor, gpi)", "zero-shot",
     ["eval500_paperfaith_armB_sd?.json"]),
]

# Static provenance notes appended to the markdown table.
NOTES = [
    "The FB-graft rows aggregate every `eval500_fbgraft_sd?_*.json` present; the "
    "previously quoted single-seed 0.064 predates sd0's JSONs landing (08-14 23:05).",
    "sd0 headline evals recorded before the P0.2 eval-seeding fix are not exactly "
    "reproducible (sd0 re-eval 0.240 vs recorded 0.318); the E2 re-eval rows are the "
    "post-fix measurement of the same checkpoints and supersede the PSMFlow headline "
    "row for comparisons.",
    "E1 oracle-aim (below) is a diagnostic, not an agent: an oracle picks among K=512 "
    "decoded prior latents using a frozen FQL expert's action.",
]


def e4_section(logs):
    """-> markdown block for E4a (fql-critic scorer) + E4b (mixture probes), or ''."""
    p = os.path.join(logs, "e4a_fql_critic_aim_cube_sd0.json")
    if not os.path.exists(p):
        return ""
    with open(p) as f:
        d = json.load(f)
    r = d["arms"]["fql_critic_aim"]
    rho = d["spearman_vs_oracle"]
    lines = ["\n## E4a: FQL-critic-as-scorer (same K=512 candidates + ODE decode as E1)\n",
             f"Success {r['success']:.3f} [{r['wilson95'][0]:.3f}, {r['wilson95'][1]:.3f}] "
             f"— below the one-step random floor (0.086). Per-step Spearman vs the oracle "
             f"ranking: mean {rho['mean']:.3f}, median {rho['median']:.3f}, "
             f"{rho['frac_above_0.3']:.0%} of steps above 0.3 — the expert's critic ranks "
             f"moderately, but argmax over 512 candidates picks "
             f"{d['picked_dist_to_expert']['mean']:.3f} from the expert action when "
             f"{d['best_available_dist']['mean']:.3f} was available. Verdict: "
             f"{d['fork_branch']}."]
    p2 = os.path.join(logs, "d1a_latent_ranking_mixhpo_ep500k.json")
    if os.path.exists(p2):
        with open(p2) as f:
            m = json.load(f)
        lines.append(f"\nE4b (mixture-trained checkpoint, 500k): ranking Spearman "
                     f"{m['spearman_mean']:.3f}, Q spread 0.86% of |Q| — same band as the "
                     f"point arm (0.10 / 1.1%) and Arm B (0.079 / 0.9%). The mixture does "
                     f"not create ranking signal.")
    return "\n".join(lines) + "\n"


def e1_section(logs):
    """-> markdown block for the E1 oracle-aim report, or '' if absent."""
    path = os.path.join(logs, "e1_oracle_aim_cube_sd0.json")
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        d = json.load(f)
    lines = ["\n## E1 oracle-aim (tools/diag_oracle_aim.py, 500 ep, K=512, ODE-100 decode)\n",
             "| Arm | Success | Wilson 95% |", "|---|---|---|"]
    for arm, r in d.get("arms", {}).items():
        lo, hi = r.get("wilson95", [None, None])
        lines.append(f"| {arm} | {r['success']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    md = d.get("aim_distance", {}).get("min_over_K", {})
    if md:
        lines.append(f"\nMean min-distance to the oracle action over K: {md['mean']:.3f} "
                     f"(p90 {md['p90']:.3f}). Verdict: {d.get('fork_branch', 'n/a')}.")
    return "\n".join(lines) + "\n"

FRACTIONS = ["10\\%", "50\\%", "100\\%"]
FRACTION = [
    ("Behavior flow (BC control)",
     ["eval500_bcflow_frac10.json"], ["eval500_bcflow_frac50.json"], ["eval500_bcflow_cube.json"]),
    ("FQL (per-task)",
     ["eval500_fql_frac10_sd?.json"], ["eval500_fql_frac50_sd?.json"], ["eval500_fql_cube_sd?.json"]),
    ("FB (zero-shot)",
     ["eval500_fb_frac10_sd?.json"], ["eval500_fb_frac50_sd?.json"], ["eval500_fb_cube_sd?.json"]),
    ("PSMFlow (zero-shot)",
     ["eval500_psmflow_frac10_sd?.json"], ["eval500_psmflow_frac50_sd?.json"],
     ["eval500_latentpsm_cube_sd?.json"]),
    ("Hybrid, deployed",
     ["eval500_hybrid_frac10_sd?.json"], ["eval500_hybrid_frac50_sd?.json"],
     ["eval500_hybrid_sd?_final.json", "eval500_hybrid_1M_sd?.json"]),
    ("Hybrid, decode-only control",
     ["eval500_hybrid_frac10_decodeonly_sd?.json"], ["eval500_hybrid_frac50_decodeonly_sd?.json"],
     ["eval500_hybrid_decodeonly_sd00?.json", "eval500_hybrid_1M_decodeonly_sd00?.json"]),
    ("Latent RL (per-task)",
     ["eval500_latrl_frac10_sd?.json"], [], ["eval500_l1stab_res0.05_*_peak*.json"]),
]


def _load(logs, patterns):
    out = []
    for pat in patterns:
        for p in sorted(glob.glob(os.path.join(logs, pat))):
            with open(p) as f:
                d = json.load(f)
            r = d.get("report", d)
            if "success" in r and r.get("success") is not None:
                out.append((os.path.basename(p), r))
    return out


def _t95(n):
    # two-sided 95% t quantiles, n-1 dof; avoids a scipy import for a 6-entry lookup
    return {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
            8: 2.365, 9: 2.306, 10: 2.262}.get(n, 1.96)


def cell(logs, patterns):
    """-> (text, dict). Mean +/- 95% CI across seeds, or a single run's Wilson interval."""
    runs = _load(logs, patterns)
    if not runs:
        return "--", {"n_seeds": 0, "files": [], "patterns": patterns}
    vals = [float(r["success"]) for _, r in runs]
    meta = {"n_seeds": len(vals), "files": [f for f, _ in runs], "values": vals,
            "modes": sorted({r.get("acting_mode", "unrecorded") for _, r in runs}),
            "epochs": sorted({r.get("restore_epoch") for _, r in runs})}
    if len(vals) == 1:
        lo, hi = runs[0][1].get("wilson95", [None, None])
        meta.update(mean=vals[0], ci_type="wilson", lo=lo, hi=hi)
        return (f"{vals[0]:.3f} [{lo:.3f}, {hi:.3f}]" if lo is not None
                else f"{vals[0]:.3f}"), meta
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    h = _t95(len(vals)) * sd / math.sqrt(len(vals))
    meta.update(mean=m, ci_type="t95_across_seeds", half_width=h)
    return f"{m:.3f} $\\pm$ {h:.3f}", meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=LOGS)
    args = ap.parse_args()
    logs = args.logs
    missing, data = [], {"headline": {}, "fraction": {}}

    head_rows = []
    for label, kind, pats in HEADLINE:
        txt, meta = cell(logs, pats)
        data["headline"][label] = {**meta, "setting": kind}
        if meta["n_seeds"] == 0:
            missing.append({"table": "headline", "row": label, "patterns": pats})
        head_rows.append((label, kind, txt, meta["n_seeds"]))

    frac_rows = []
    for label, *cols in FRACTION:
        texts, metas = [], []
        for frac, pats in zip(FRACTIONS, cols):
            txt, meta = cell(logs, pats)
            texts.append(txt)
            metas.append(meta)
            data["fraction"].setdefault(label, {})[frac.replace("\\", "")] = meta
            if meta["n_seeds"] == 0 and pats:
                missing.append({"table": "fraction", "row": label, "col": frac,
                                "patterns": pats})
        frac_rows.append((label, texts, metas))

    os.makedirs(os.path.join(REPO, "PAPER/ICLR/tables"), exist_ok=True)
    os.makedirs(os.path.join(REPO, "docs/tables"), exist_ok=True)

    with open(os.path.join(REPO, "PAPER/ICLR/tables/table_headline.tex"), "w") as f:
        f.write("% generated by tools/make_tables.py -- do not edit by hand\n")
        f.write("\\begin{tabular}{llr}\n\\toprule\nMethod & Setting & "
                "Success (500 ep) \\\\\n\\midrule\n")
        for label, kind, txt, n in head_rows:
            f.write(f"{label} & {kind} & {txt} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    with open(os.path.join(REPO, "PAPER/ICLR/tables/table_fraction.tex"), "w") as f:
        f.write("% generated by tools/make_tables.py -- do not edit by hand\n")
        f.write("\\begin{tabular}{lrrr}\n\\toprule\nMethod & " +
                " & ".join(FRACTIONS) + " \\\\\n\\midrule\n")
        for label, texts, _ in frac_rows:
            f.write(f"{label} & " + " & ".join(texts) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    def md(t):
        return t.replace("$\\pm$", "±")

    with open(os.path.join(REPO, "docs/tables/results.md"), "w") as f:
        f.write("# Results tables (generated by `tools/make_tables.py`)\n\n"
                "cube-single-play-singletask-v0. 500-episode evals. Multiple seeds are "
                "mean ± 95% CI across seeds; a single seed shows its Wilson interval.\n\n"
                "## Headline\n\n| Method | Setting | Success (500 ep) | seeds |\n"
                "|---|---|---|---|\n")
        for label, kind, txt, n in head_rows:
            f.write(f"| {label} | {kind} | {md(txt)} | {n} |\n")
        f.write("\n## Data fraction\n\n| Method | " + " | ".join(
            x.replace("\\", "") for x in FRACTIONS) + " |\n|---|---|---|---|\n")
        for label, texts, _ in frac_rows:
            f.write(f"| {label} | " + " | ".join(md(t) for t in texts) + " |\n")
        f.write(e1_section(logs))
        f.write(e4_section(logs))
        f.write("\n## Provenance notes\n\n")
        for note in NOTES:
            f.write(f"- {note}\n")
        if missing:
            f.write("\n## Cells with no data yet\n\n")
            for m in missing:
                f.write(f"- {m['table']}: {m['row']}"
                        + (f" @ {m['col']}" if "col" in m else "")
                        + f"  (`{'`, `'.join(m['patterns'])}`)\n")

    data["missing"] = missing
    with open(os.path.join(logs, "table_dataset_fraction.json"), "w") as f:
        json.dump(data, f, indent=2)

    print(open(os.path.join(REPO, "docs/tables/results.md")).read())
    print(f"\n{len(missing)} empty cell(s); wrote PAPER/ICLR/tables/*.tex, "
          f"docs/tables/results.md, {logs}/table_dataset_fraction.json")


if __name__ == "__main__":
    main()
