"""Does best-of-64 acting pick a different kind of action than a random BC draw? (2026-10-13)

Reads the npz files written by tools/diag_psmgoal_rollout_ranking.py (every eval step: the 64
decoded candidate actions `actions_cand` and the executed index `chosen`). Per step, over the 64
candidate actions:

  spread         mean over action dims of the std across the 64 candidates
  separation     2-means on the 64 actions (best of 5 inits): distance between the two centres
                 divided by the pooled within-group std. Near 1-2 means one blob; well above 3
                 means two distinct groups
  minor_share    share of the 64 in the smaller group
  pick_minor     the pick is in the smaller group (compare with minor_share, the rate for a
                 random draw)
  pick_typ_rank  rank of the pick's distance to the candidates' mean action among the 64
                 (1 = most typical, 64 = most unusual, chance 32.5)
  grip_flip      the pick's gripper command (last dim) has the opposite sign to the majority of
                 the 64; grip_flip_rand is the same rate for a random candidate

Steps where separation > SEP_MULTI count as "two groups"; pick_minor is also reported on those
steps alone. Results are split by successful and failed episodes.

Run (CPU):
  .venv/bin/python tools/diag_pick_modes.py --report-out $PSM_DATA/logs/diag_pick_modes.json \
    $PSM_DATA/logs/rollout_candidates/fbc3pt_sd000_500k_task*.npz
"""
import argparse
import json
import os

import numpy as np

SEP_MULTI = 3.0


def two_means(x, rng, inits=5, iters=20):
    """2-means on x (N, d). Returns labels (N,) of the lowest-inertia run."""
    best, best_lab = np.inf, None
    for _ in range(inits):
        c = x[rng.choice(x.shape[0], 2, replace=False)].copy()
        for _ in range(iters):
            lab = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
            for j in range(2):
                if np.any(lab == j):
                    c[j] = x[lab == j].mean(0)
        inertia = ((x - c[lab]) ** 2).sum()
        if inertia < best:
            best, best_lab = inertia, lab
    return best_lab


def step_metrics(acts, pick, rng):
    """acts (64, d) candidate actions, pick int."""
    acts = np.asarray(acts, np.float64)
    n = acts.shape[0]
    lab = two_means(acts, rng)
    sizes = np.bincount(lab, minlength=2)
    minor = int(sizes.argmin())
    if sizes.min() >= 2:
        cs = np.stack([acts[lab == j].mean(0) for j in range(2)])
        within = np.sqrt(np.mean([((acts[lab == j] - cs[j]) ** 2).sum(-1).mean() for j in range(2)]))
        sep = float(np.linalg.norm(cs[0] - cs[1]) / max(within, 1e-8))
    else:
        sep = 0.0
    d_mean = np.linalg.norm(acts - acts.mean(0), axis=1)
    typ_rank = 1.0 + (d_mean < d_mean[pick]).sum() + 0.5 * ((d_mean == d_mean[pick]).sum() - 1)
    g = np.sign(acts[:, -1])
    maj = 1.0 if (g >= 0).sum() >= n / 2 else -1.0
    return {
        'spread': float(acts.std(0).mean()),
        'separation': sep,
        'minor_share': float(sizes.min() / n),
        'pick_minor': float(lab[pick] == minor),
        'pick_typ_rank': float(typ_rank),
        'grip_flip': float(g[pick] != maj),
        'grip_flip_rand': float((g != maj).mean()),
    }


def summarise(rows):
    if not rows:
        return {'n_steps': 0}
    keys = rows[0].keys()
    out = {'n_steps': len(rows)}
    out.update({k: float(np.mean([r[k] for r in rows])) for k in keys})
    multi = [r for r in rows if r['separation'] > SEP_MULTI]
    out['multi_share'] = len(multi) / len(rows)
    if multi:
        out['multi_pick_minor'] = float(np.mean([r['pick_minor'] for r in multi]))
        out['multi_minor_share'] = float(np.mean([r['minor_share'] for r in multi]))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('npz', nargs='+')
    p.add_argument('--report-out', required=True)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    report = {'what': __doc__.split('\n\n')[0], 'files': {}, 'sep_multi': SEP_MULTI}
    pooled = {'success': [], 'fail': []}
    for path in args.npz:
        z = np.load(path)
        succ = z['ep_success'][z['episode']]
        per = {'success': [], 'fail': []}
        for i in range(z['chosen'].shape[0]):
            m = step_metrics(z['actions_cand'][i], int(z['chosen'][i]), rng)
            per['success' if succ[i] else 'fail'].append(m)
        report['files'][os.path.basename(path)] = {k: summarise(v) for k, v in per.items()}
        for k, v in per.items():
            pooled[k] += v
        print(os.path.basename(path), {k: summarise(v).get('n_steps') for k, v in per.items()}, flush=True)
    report['pooled'] = {k: summarise(v) for k, v in pooled.items()}
    report['pooled']['all'] = summarise(pooled['success'] + pooled['fail'])
    os.makedirs(os.path.dirname(os.path.abspath(args.report_out)), exist_ok=True)
    with open(args.report_out, 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report['pooled'], indent=2))


if __name__ == '__main__':
    main()
