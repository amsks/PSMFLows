#!/usr/bin/env python3
"""Score a training arm on LADDER STABILITY, not just its pooled mean.

An arm that scores 0.45 flat beats one that scores 0.45 by averaging 0.70 and 0.09,
because only the first is shippable. See docs/design/2026-09-08-oscillation-stability.md.

Per seed, over the checkpoints in [--lo, --hi]:

  swing   max - min of the 500-episode ladder
  step    mean |difference| between consecutive checkpoints
  mean    pooled mean, reported beside the two above and never instead of them

Reads 500-episode report JSONs (eval500 output) when present, and falls back to the run's
own in-loop eval.csv with a loud warning -- the in-loop 50-episode points swing +/-0.15 and
are not a substitute.

Usage:
    .venv/bin/python tools/stability_ladder.py --exp $PSM_DATA/exp/PSMFLows \
        --groups affine_strict_cube,tau1e3_cube,oc1e4_cube --logs $PSM_DATA/logs
"""

import argparse
import csv
import glob
import json
import os
import statistics as st

#: An eval carrying any of these as a CLI override was deliberately run OFF the run's own
#: config -- a gpi_num_u sweep cell, a selection ablation, a different box. Those share the
#: run directory and checkpoint with the run's own eval and would otherwise silently
#: overwrite it, since the ladder is keyed by restore_epoch alone.
ACTING_OVERRIDES = frozenset({
    "gpi_num_u", "gpi_select", "gpi_topm", "gpi_index_seed", "gpi_prior_shrink",
    "u_clip", "index_clip", "acting", "actor_mode", "index_agg", "gpi_decode",
})


def run_env(seed_dir):
    """The env the run was TRAINED on, from its own flags.json."""
    try:
        with open(os.path.join(seed_dir, "flags.json")) as fh:
            return json.load(fh).get("env_name")
    except (OSError, ValueError):
        return None


def ladder_from_eval500(logs, group, seed_dir, min_episodes=100):
    """500-episode points for one run, as {step: success}, or {} if none are on disk.

    Three filters, each of which cost a wrong table before it was added. The ladder is keyed
    by restore_epoch, so anything sharing a (run, epoch) silently overwrites the real entry:

    - acting overrides -- a gpi_num_u sweep cell or a selection ablation is a different
      POLICY on the same weights;
    - a different TASK -- `..._task3_sd0.json` is the same checkpoint scored on another
      reward and carries no CLI override to distinguish it, only a different `env`;
    - smokes -- `smoke_*` reports are a handful of episodes and read like a real number.
    """
    out = {}
    run = os.path.basename(seed_dir)
    want_env = run_env(seed_dir)
    for f in glob.glob(os.path.join(logs, "*.json")):
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        acs = d.get("agent_config_source") or {}
        src = str(acs.get("flags_json", "") or d.get("restore_path", ""))
        if f"/{group}/" not in src or run not in src:
            continue
        if ACTING_OVERRIDES & set(acs.get("cli_overrides") or ()):
            continue
        if want_env and d.get("env") and d["env"] != want_env:
            continue
        n_ep = len(d.get("per_episode_success") or ())
        if n_ep and n_ep < min_episodes:
            continue
        ep, ok = d.get("restore_epoch"), d.get("success")
        if ep is not None and ok is not None:
            out[int(ep)] = float(ok)
    return out


def ladder_from_eval_csv(seed_dir):
    path = os.path.join(seed_dir, "eval.csv")
    if not os.path.isfile(path):
        return {}
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    key = next((c for c in rows[0] if "success" in c.lower()), None)
    if key is None:
        return {}
    out = {}
    for r in rows:
        try:
            out[int(float(r["step"]))] = float(r[key])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def score(ladder, lo, hi):
    pts = sorted((s, v) for s, v in ladder.items() if lo <= s <= hi)
    if len(pts) < 2:
        return None
    vals = [v for _, v in pts]
    steps = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
    return {"n": len(vals), "mean": st.mean(vals), "swing": max(vals) - min(vals),
            "step": st.mean(steps), "min": min(vals), "max": max(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--logs", default="")
    ap.add_argument("--groups", required=True, help="comma-separated run_group names")
    ap.add_argument("--lo", type=int, default=300000)
    ap.add_argument("--hi", type=int, default=500000)
    ap.add_argument("--report_out", default="")
    a = ap.parse_args()

    report = {"window": [a.lo, a.hi], "groups": {}}
    print(f"{'group':26s}{'seed':7s}{'src':8s}{'n':>3s}{'mean':>8s}{'swing':>8s}{'step':>8s}"
          f"{'min':>8s}{'max':>8s}")
    for g in a.groups.split(","):
        g = g.strip()
        rows = []
        for sd in sorted(glob.glob(os.path.join(a.exp, g, "*/"))):
            sd = sd.rstrip("/")
            lad, src = {}, "e500"
            if a.logs:
                lad = ladder_from_eval500(a.logs, g, sd)
            if len(lad) < 2:
                lad, src = ladder_from_eval_csv(sd), "csv*"
            s = score(lad, a.lo, a.hi)
            if s is None:
                continue
            s["src"] = src
            rows.append((os.path.basename(sd)[:6], s))
            print(f"{g[:26]:26s}{os.path.basename(sd)[:6]:7s}{src:8s}{s['n']:3d}"
                  f"{s['mean']:8.3f}{s['swing']:8.3f}{s['step']:8.3f}{s['min']:8.3f}{s['max']:8.3f}")
        if rows:
            report["groups"][g] = {k: v for k, v in rows}
            print(f"{'':26s}{'ARM':7s}{'':8s}{'':3s}"
                  f"{st.mean(r[1]['mean'] for r in rows):8.3f}"
                  f"{st.mean(r[1]['swing'] for r in rows):8.3f}"
                  f"{st.mean(r[1]['step'] for r in rows):8.3f}\n")
    if any(v["src"] == "csv*" for gr in report["groups"].values() for v in gr.values()):
        print("csv* = in-loop 50-episode evals, which swing +/-0.15. NOT reportable; "
              "run scripts/eval500.sh before drawing a conclusion.")
    if a.report_out:
        os.makedirs(os.path.dirname(a.report_out), exist_ok=True)
        with open(a.report_out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"report -> {a.report_out}")


if __name__ == "__main__":
    main()
