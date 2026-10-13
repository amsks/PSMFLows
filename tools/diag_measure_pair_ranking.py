"""Do two psmgoal measures order the same BC draws the same way? (2026-10-10, task 2b of
mailbox/tasks/2026-10-09-design-decisions.md)

Two groups trained identically except for the bootstrap latent (default: the flowbc actor run
psmgoal_ja_fbc3_pt_cube and the run whose actor is pinned to BC, psmgoal_ja_fbcBC_pt_cube).
Same seed and step for both. On the fixed rows, reference states, hindsight goals and
candidate table of tools/diag_actor_ranking.py (column 0 = the data's own latent, columns 1..63
= clipped prior draws, i.e. BC draws), each measure scores every candidate with its own h(g)
(the per-goal score of `_hgoal_each_score`). Per k, over the 63 BC draws at each state:

  spearman      rank correlation of the two score rows (mean over states)
  top1_agree    both pick the same draw
  top8_overlap  share of A's top 8 that are also in B's top 8 (chance 8/63)
  a_best_in_b   rank of A's argmax in B's order (1 = best, chance 32)
  b_best_in_a   rank of B's argmax in A's order
  data rank     each measure's rank of the data latent among the 63 draws (chance 32.5)
  spread        each measure's within-state score std / std over states of the best score

Run (GPU):
  .venv/bin/python tools/diag_measure_pair_ranking.py \
    --report-out $PSM_DATA/logs/diag_measure_pair_ranking.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax
from tools.diag_actor_ranking import (
    KS,
    PSM_DATA,
    find_run,
    goal_rows,
    load_sample,
    make_scorer,
    rank_among,
    rank_of_first,
    restore,
    spread_ratio,
)

TOP = 8


def spearman_rows(a, b):
    """Per-row Spearman correlation of a (N, C) and b (N, C), average ranks on ties."""
    from scipy.stats import rankdata
    ra = rankdata(a, axis=1)
    rb = rankdata(b, axis=1)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    return np.where(den > 0, (ra * rb).sum(axis=1) / np.where(den > 0, den, 1.0), 0.0)


def topk_overlap(a, b, k=TOP):
    """Share of a's top-k columns that are also in b's top-k, per row."""
    ia = np.argsort(-a, axis=1)[:, :k]
    ib = np.argsort(-b, axis=1)[:, :k]
    return np.array([len(set(x) & set(y)) / k for x, y in zip(ia, ib)])


def pair_metrics(sa, sb, top=TOP):
    """sa, sb (N, 64): column 0 = data latent, 1.. = BC draws, scored by measure A and B."""
    sa = np.asarray(sa, np.float64)
    sb = np.asarray(sb, np.float64)
    da, db = sa[:, 1:], sb[:, 1:]
    n = np.arange(da.shape[0])
    a_best, b_best = da.argmax(axis=1), db.argmax(axis=1)
    c = da.shape[1]
    return {
        'n': int(da.shape[0]),
        'spearman': float(spearman_rows(da, db).mean()),
        'top1_agree': float((a_best == b_best).mean()),
        'top1_chance': 1.0 / c,
        'top8_overlap': float(topk_overlap(da, db, top).mean()),
        'top8_chance': top / c,
        'a_best_in_b': float(rank_among(db, db[n, a_best]).mean()),
        'b_best_in_a': float(rank_among(da, da[n, b_best]).mean()),
        'rank_chance': (c + 1) / 2.0,
        'a_data_rank': float(rank_of_first(sa).mean()),
        'b_data_rank': float(rank_of_first(sb).mean()),
        'a_spread': spread_ratio(sa),
        'b_spread': spread_ratio(sb),
    }


def table(cells):
    """Mean over seeds per (step, k), with the seed standard error."""
    agg = {}
    for c in cells:
        for k, m in c['per_k'].items():
            agg.setdefault((c['step'], int(k)), []).append(m)
    rows = []
    for (s, k), ms in sorted(agg.items()):
        row = {'step': s, 'k': k, 'n_seeds': len(ms)}
        for key in ms[0]:
            vals = [m[key] for m in ms]
            row[key] = float(np.mean(vals))
            if len(vals) > 1:
                row[key + '_se'] = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        rows.append(row)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--group-a', default='psmgoal_ja_fbc3_pt_cube')
    p.add_argument('--group-b', default='psmgoal_ja_fbcBC_pt_cube')
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--steps', nargs='+', type=int, default=[250000, 500000])
    p.add_argument('--ks', nargs='+', type=int, default=list(KS))
    p.add_argument('--n-states', type=int, default=2000)
    p.add_argument('--sample-seed', type=int, default=0)
    p.add_argument('--chunk', type=int, default=16)
    p.add_argument('--preimages', default=os.path.join(PSM_DATA, 'preimages/cube-single-play.npz'))
    p.add_argument('--report-out', default=os.path.join(PSM_DATA, 'logs/diag_measure_pair_ranking.json'))
    args = p.parse_args(argv)

    t0 = time.time()
    data, end_of_row, rows, ref_rows, prior = load_sample(args.preimages, args.n_states, args.sample_seed)
    refs = data['next_observations'][ref_rows].astype(np.float32)
    obs = data['observations'][rows].astype(np.float32)
    ex_obs, ex_act = data['observations'][:1], data['actions'][:1]
    report = {'what': __doc__.split('\n\n')[0], 'args': vars(args), 'cells': []}
    for seed in args.seeds:
        runs = {'a': find_run(args.group_a, seed), 'b': find_run(args.group_b, seed)}
        for step in args.steps:
            if not all(os.path.exists(os.path.join(r, f'params_{step}.pkl')) for r in runs.values()):
                print(f'SKIP sd{seed} {step}: missing checkpoint', flush=True)
                continue
            scores = {}
            for side, run_dir in runs.items():
                agent, config = restore(run_dir, step, ex_obs, ex_act)
                u_clip = float(config['u_clip'])
                assert bool(config.get('use_point_preimage', False)), 'u_data is the point preimage'
                u_data = np.clip(data['noise_preimage_point'][rows], -u_clip, u_clip).astype(np.float32)
                cand = np.concatenate([u_data[:, None], np.clip(prior, -u_clip, u_clip)], axis=1)
                scorer, _, _ = make_scorer(agent, config, args.chunk)
                scores[side] = {}
                for k in args.ks:
                    ok, grow = goal_rows(rows, end_of_row, k)
                    g = data['next_observations'][grow[ok]].astype(np.float32)
                    scores[side][k] = scorer(obs[ok], cand[ok], g, refs, prior[ok, 0])[0]
            cell = {'seed': seed, 'step': step, 'runs': runs, 'per_k': {}}
            for k in args.ks:
                m = pair_metrics(scores['a'][k], scores['b'][k])
                cell['per_k'][str(k)] = m
                print(f'sd{seed} {step} k={k}: spearman {m["spearman"]:.3f} top1 {m["top1_agree"]:.3f} '
                      f'top8 {m["top8_overlap"]:.3f} a_best_in_b {m["a_best_in_b"]:.1f} '
                      f'b_best_in_a {m["b_best_in_a"]:.1f} data rank a {m["a_data_rank"]:.1f} '
                      f'b {m["b_data_rank"]:.1f}', flush=True)
            report['cells'].append(cell)
            report['table'] = table(report['cells'])
            report['elapsed_s'] = time.time() - t0
            os.makedirs(os.path.dirname(os.path.abspath(args.report_out)), exist_ok=True)
            with open(args.report_out, 'w') as f:
                json.dump(report, f, indent=2)
    print('report ->', args.report_out, f'({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
