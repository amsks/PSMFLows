"""Table of the five-task means from diag_critic_compare JSONs. Usage: python aggregate.py <json> ..."""
import json
import os
import sys

ROWS = [("index_heldout", "pearson"), ("index_heldout", "auc"),
        ("measure", "h1", "pct_true_future_u_data"), ("measure", "h10", "pct_true_future_u_data"),
        ("measure", "h50", "pct_true_future_u_data"), ("measure", "h1", "pct_u_data_minus_random"),
        ("measure", "h5", "pct_u_data_minus_random"), ("measure", "logit_std_over_pool"),
        ("ranking", "within_over_across"), ("ranking", "pooled_spearman_q_vs_progress_cube"),
        ("ranking", "pick_progress_cube"), ("ranking", "rand_progress_cube"),
        ("slope", "slope_over_spread"), ("slope", "q_over_bc_term"), ("slope", "ascent_gain_over_spread"),
        ("slope", "actor_minus_bc_q_over_spread"), ("slope", "actor_shift"), ("slope", "actor_percentile"),
        ("slope", "bc_percentile"), ("slope", "actor_progress_cube"), ("slope", "bc_progress_cube")]


def get(d, path):
    for k in path:
        d = d[k]
    return d


files = sys.argv[1:]
reps = [json.load(open(f))["mean_over_tasks"] for f in files]
names = [os.path.basename(f).replace(".json", "") for f in files]
print("| metric | " + " | ".join(names) + " |")
print("|---" * (len(names) + 1) + "|")
for r in ROWS:
    print("| " + ".".join(r) + " | " + " | ".join(f"{get(x, r):.4g}" for x in reps) + " |")
