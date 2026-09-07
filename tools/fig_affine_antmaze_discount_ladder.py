"""Antmaze H2: the 500-episode checkpoint ladder at gamma = 0.98 vs 0.99 vs 0.995.

Pre-registration and verdict: `docs/design/2026-09-06-antmaze-failure-tests.md` (H2). The
repo-default affine-strict agent is null on antmaze-medium at the default `discount=0.98`
(late-checkpoint mean 0.081 +/- 0.050 against a BC control of 0.072, episodes timing out at
1000 steps). H2 said the ~50-step effective horizon `1/(1-0.98)` is too short for a maze
whose goal is hundreds of steps away. These runs raise it to 100 (`0.99`) and 200
(`0.995`), three seeds each, everything else byte-identical.

Companion of `tools/fig_affine_cube_ladder.py`: same JSON schema (runs / note / series /
eval500 / aggregate), same per-cell Wilson intervals, same "the SHAPE of the series is the
point, not its endpoint" framing. The difference is that the object plotted is the
DISCOUNT SWEEP -- three arms, never one -- because the pre-registered signature (monotone
in the horizon, best at 500k) is not what happened: 0.995 wins early and then collapses,
so any single-checkpoint reading of this experiment is wrong in a different direction for
each arm.

`tools/fig_affine_discount_ladder.py` is a different figure by a different study: it takes
this result to CUBE and POINTMAZE. This one is the antmaze evidence it cites.

IDEMPOTENT AND PARTIAL-SAFE, by design: it reads whatever `eval500_*.json` cells exist and
renders the rest as gaps, so it can be re-run at any time as queued SLURM evals land and
simply fills in. It prints a coverage line (cells found / cells expected) so a re-run
announces whether the ladder is complete. Re-running never needs any argument but the two
paths; nothing here writes to $PSM_DATA.

Outputs (docs/figures + docs/tables -- lab-notebook artifacts, NOT PAPER/ICLR/figures):
    docs/figures/<name>.png
    docs/figures/<name>.json
    docs/tables/affine_antmaze_discount_ladder.md

Usage:
    .venv/bin/python tools/fig_affine_antmaze_discount_ladder.py --logs $PSM_DATA/logs \
        --exp $PSM_DATA/exp/PSMFLows --name 2026-09-07-affine-antmaze-discount-ladder
"""
import argparse
import csv
import glob
import json
import os
import statistics as st

import matplotlib

matplotlib.use('Agg')

import figstyle
import matplotlib.pyplot as plt

EPOCHS = list(range(50000, 550000, 50000))
BC_CONTROL = 0.072  # antmaze, 500 ep, frozen flow acting alone (agent=fql bc_only=true)
# The affine LATENT-ACTOR arm on the same env/flow/checkpoints -- the only pre-H2 antmaze
# arm above the BC control, and the bar a discount fix has to clear to be interesting.
ACTOR_ARM = 0.21

# One entry per discount. `group` is the run_group under --exp; `pat` is the report
# basename with {n} the epoch in thousands and {s} the seed. gamma=0.98 is the ORIGINAL
# affine_strict_antmaze batch, whose basenames carry no discount token; its ladder is
# partial by history (it was never run at every checkpoint) and the table shows the gaps
# rather than filling them.
ARMS = [
    ('0.98', {'group': 'affine_strict_antmaze',
              'pat': 'eval500_affine{n}k_strict_antmaze_sd{s}.json',
              'label': r'$\gamma$=0.98 (default)'}),
    ('0.99', {'group': 'affine_strict_antmaze_g99',
              'pat': 'eval500_affine{n}k_strict_antmaze_g99_sd{s}.json',
              'label': r'$\gamma$=0.99'}),
    ('0.995', {'group': 'affine_strict_antmaze_g995',
               'pat': 'eval500_affine{n}k_strict_antmaze_g995_sd{s}.json',
               'label': r'$\gamma$=0.995'}),
]

TRAIN_COLS = ['w_enc_spread', 'psi_q_index_spread_rel', 'psi_q_spread_rel',
              'psi_q_range_rel', 'psi_q_spread', 'psm_loss', 'psm_diag', 'psm_offdiag',
              'orth_loss', 'orth_diag', 'orth_offdiag']
EVAL_COLS = {'eval_success': 'success', 'eval_episode_length': 'episode.length'}


def _read_csv(path, prefix, cols):
    """{column: [values]} plus 'step', for the subset of `cols` the csv actually has."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {'step': [int(float(r['step'])) for r in rows]}
    for name, col in cols.items():
        key = prefix + col
        if key in rows[0]:
            out[name] = [float(r[key]) for r in rows]
    return out


def collect(logs, exp, seeds):
    """-> runs, series, eval500, each keyed by discount then by seed."""
    runs, series, eval500 = {}, {}, {}
    for gamma, spec in ARMS:
        runs[gamma], series[gamma], eval500[gamma] = {}, {}, {}
        for s in seeds:
            hits = sorted(glob.glob(os.path.join(exp, spec['group'], f'sd{int(s):03d}_*')))
            if hits:
                run = hits[-1]
                runs[gamma][str(s)] = run
                tr = _read_csv(os.path.join(run, 'train.csv'), 'training/',
                               {c: c for c in TRAIN_COLS})
                ev = _read_csv(os.path.join(run, 'eval.csv'), 'evaluation/', EVAL_COLS)
                ser = {'train_steps': tr.pop('step', []), 'eval_steps': ev.pop('step', [])}
                ser.update(ev)
                ser.update(tr)
                series[gamma][str(s)] = ser

            cells = {}
            for ep in EPOCHS:
                p = os.path.join(logs, spec['pat'].format(n=ep // 1000, s=s))
                if not os.path.exists(p):
                    continue
                with open(p) as fh:
                    d = json.load(fh)
                # A mis-typed RESTORE_EPOCH would otherwise land a 100k number in the 500k
                # row -- the exact failure the epoch token in the basename exists to stop.
                if int(d['restore_epoch']) != ep:
                    raise SystemExit(f'{p}: restore_epoch {d["restore_epoch"]} != {ep}')
                k, n = int(d['num_success']), int(d['num_episodes'])
                _, lo, hi = figstyle.wilson(k, n)
                cells[str(ep)] = {'success': float(d['success']),
                                  'wilson95': [round(lo, 4), round(hi, 4)],
                                  'num_success': k, 'num_episodes': n,
                                  'json': os.path.basename(p),
                                  'acting': d.get('acting'),
                                  'acting_mode': d.get('acting_mode'),
                                  'num_workers': d.get('num_workers')}
            eval500[gamma][str(s)] = cells
    return runs, series, eval500


def aggregate(eval500, seeds):
    per_arm = {}
    for gamma, _ in ARMS:
        per_epoch = {}
        for ep in EPOCHS:
            vals = [eval500[gamma][str(s)][str(ep)]['success'] for s in seeds
                    if str(ep) in eval500[gamma].get(str(s), {})]
            if not vals:
                continue
            per_epoch[str(ep)] = {'n_seeds': len(vals), 'mean': round(st.mean(vals), 4),
                                  'std': round(st.stdev(vals), 4) if len(vals) > 1 else 0.0,
                                  'values': vals}

        def pooled(lo, hi, _g=gamma):
            vals = [c['success'] for s in seeds
                    for e, c in eval500[_g].get(str(s), {}).items()
                    if lo <= int(e) <= hi]
            if not vals:
                return None
            m, hw = figstyle.mean_ci(vals)
            return {'from_epoch': lo, 'to_epoch': hi, 'n_measurements': len(vals),
                    'mean': round(m, 4),
                    'std': round(st.stdev(vals), 4) if len(vals) > 1 else 0.0,
                    'ci95_half_width': round(hw, 4),
                    'min': round(min(vals), 4), 'max': round(max(vals), 4)}

        per_arm[gamma] = {'per_epoch': per_epoch,
                          'early_50k_250k': pooled(50000, 250000),
                          'late_300k_500k': pooled(300000, 500000),
                          'all': pooled(0, 10**9)}
    return {'arms': per_arm, 'bc_control': BC_CONTROL, 'actor_arm': ACTOR_ARM}


def write_table(eval500, agg, seeds, out_md):
    L = []
    L.append('# Affine strict on antmaze-medium-navigate: the discount ladder')
    L.append('')
    L.append('Arm: `psi_form=affine policy_index=latent train_actor=false acting=gpi`')
    L.append('(the repo default `agent=psmflow`), three seeds per discount, everything')
    L.append('except `agent.discount` byte-identical across the three arms.')
    L.append('')
    L.append('GENERATED by `tools/fig_affine_antmaze_discount_ladder.py`; do not hand-edit.')
    L.append('Pre-registration and verdict: `docs/design/2026-09-06-antmaze-failure-tests.md`.')
    L.append('')
    L.append('Every cell is one 500-episode `tools/eval_checkpoint.py` run (serial path,')
    L.append('`eval_workers=1`), success with its Wilson 95% interval. Effective horizon is')
    L.append('`1/(1-gamma)`: 50, 100 and 200 steps against a 1000-step episode.')
    L.append('')
    for gamma, spec in ARMS:
        L.append(f'## gamma = {gamma}  (horizon {round(1 / (1 - float(gamma)))} steps)')
        L.append('')
        L.append('| epoch | ' + ' | '.join(f'seed {s}' for s in seeds)
                 + ' | mean +/- std (across seeds) |')
        L.append('|---' * (len(seeds) + 2) + '|')
        for ep in EPOCHS:
            cells = []
            for s in seeds:
                c = eval500[gamma].get(str(s), {}).get(str(ep))
                cells.append('—' if c is None else
                             f"{c['success']:.3f} [{c['wilson95'][0]:.3f}, {c['wilson95'][1]:.3f}]")
            a = agg['arms'][gamma]['per_epoch'].get(str(ep))
            agg_cell = '—' if a is None else f"{a['mean']:.3f} +/- {a['std']:.3f} (n={a['n_seeds']})"
            L.append(f'| {ep // 1000}k | ' + ' | '.join(cells) + f' | {agg_cell} |')
        L.append('')
    L.append('## Pooled windows (each (checkpoint, seed) 500-episode cell is one draw)')
    L.append('')
    L.append('| arm | window | n | mean | std | 95% CI half-width | min | max |')
    L.append('|---|---|---|---|---|---|---|---|')
    for gamma, _ in ARMS:
        for key, label in (('early_50k_250k', '50k-250k'), ('late_300k_500k', '300k-500k'),
                           ('all', 'all measured')):
            b = agg['arms'][gamma].get(key)
            if b:
                L.append(f"| gamma={gamma} | {label} | {b['n_measurements']} | {b['mean']:.3f} | "
                         f"{b['std']:.3f} | +/-{b['ci95_half_width']:.3f} | "
                         f"{b['min']:.3f} | {b['max']:.3f} |")
    L.append(f"| BC control (frozen flow alone) | — | — | {agg['bc_control']:.3f} "
             '| — | — | — | — |')
    L.append(f"| affine latent-actor arm (same env/flow) | — | — | ~{agg['actor_arm']:.2f} "
             '| — | — | — | — |')
    L.append('')
    L.append('The gamma=0.98 ladder is partial by history -- that batch was never evaluated')
    L.append('at every checkpoint -- so its gaps are shown rather than filled. The gamma=0.995')
    L.append('ladder stops at 300k plus a 500k endpoint: its in-loop success is exactly 0.000')
    L.append('on all three seeds from 300k onward, and the 300k and 500k cells are there to')
    L.append('date that collapse at 500 episodes rather than at 50.')
    L.append('')
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, 'w') as f:
        f.write('\n'.join(L))
    print(f'wrote {out_md}')


def plot(series, eval500, agg, seeds, out_png):
    figstyle.use_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))

    ax = axes[0]
    for i, (gamma, spec) in enumerate(ARMS):
        col = figstyle.PALETTE[i]
        # Individual (checkpoint, seed) cells, faint: the spread between seeds at one
        # checkpoint is part of the result, not something the mean should hide.
        for s in seeds:
            cells = eval500[gamma].get(str(s), {})
            xs = [ep for ep in EPOCHS if str(ep) in cells]
            if xs:
                ax.plot([x / 1000 for x in xs], [cells[str(ep)]['success'] for ep in xs],
                        marker='.', ls=':', lw=0.7, ms=3, color=col, alpha=0.45)
        pe = agg['arms'][gamma]['per_epoch']
        xs = [ep for ep in EPOCHS if str(ep) in pe]
        if not xs:
            continue
        ys = [pe[str(ep)]['mean'] for ep in xs]
        er = [pe[str(ep)]['std'] for ep in xs]
        ax.errorbar([x / 1000 for x in xs], ys, yerr=er, marker='o', capsize=2, lw=1.6,
                    color=col, label=spec['label'])
    ax.axhline(BC_CONTROL, color=figstyle.INK_MUTED, ls='--', lw=1.0)
    ax.text(495, BC_CONTROL + 0.02, 'BC control 0.072', color=figstyle.INK_MUTED,
            fontsize=6, ha='right')
    ax.axhline(ACTOR_ARM, color=figstyle.INK_MUTED, ls=':', lw=1.0)
    ax.text(495, ACTOR_ARM + 0.02, 'latent-actor arm ~0.21', color=figstyle.INK_MUTED,
            fontsize=6, ha='right')
    ax.set_xlabel('checkpoint (k steps)')
    ax.set_ylabel('success (500 episodes)')
    ax.set_title('antmaze: 500-ep ladder by discount')
    ax.set_ylim(-0.02, 0.85)
    ax.legend(fontsize=6)

    ax = axes[1]
    for i, (gamma, spec) in enumerate(ARMS):
        col = figstyle.PALETTE[i]
        for j, s in enumerate(seeds):
            ser = series[gamma].get(str(s), {})
            if ser.get('eval_steps') and ser.get('eval_success'):
                ax.plot([x / 1000 for x in ser['eval_steps']], ser['eval_success'],
                        marker='.', lw=0.9, color=col, alpha=0.75,
                        label=spec['label'] if j == 0 else None)
    ax.axhline(BC_CONTROL, color=figstyle.INK_MUTED, ls='--', lw=1.0)
    ax.set_xlabel('step (k)')
    ax.set_ylabel('success (50 episodes, in-loop)')
    ax.set_title('in-loop eval (noisy: 95% CI ~ +/-0.115)')
    ax.set_ylim(-0.02, 0.85)
    ax.legend(fontsize=6)

    ax = axes[2]
    for i, (gamma, spec) in enumerate(ARMS):
        col = figstyle.PALETTE[i]
        for j, s in enumerate(seeds):
            ser = series[gamma].get(str(s), {})
            if ser.get('train_steps') and ser.get('w_enc_spread'):
                n = min(len(ser['train_steps']), len(ser['w_enc_spread']))
                ax.plot([x / 1000 for x in ser['train_steps'][:n]],
                        ser['w_enc_spread'][:n], lw=0.9, color=col, alpha=0.75,
                        label=spec['label'] if j == 0 else None)
    ax.set_xlabel('step (k)')
    ax.set_ylabel('mean pairwise ||w(u_i) - w(u_j)||')
    ax.set_title('policy-encoder spread (collapse -> 0)')
    ax.legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f'wrote {out_png}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', required=True, help='dir of eval500_*.json reports')
    ap.add_argument('--exp', required=True,
                    help='experiment root holding affine_strict_antmaze{,_g99,_g995}/')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--name', default='2026-09-07-affine-antmaze-discount-ladder')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--table', default='affine_antmaze_discount_ladder.md')
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(',')]
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    runs, series, eval500 = collect(args.logs, args.exp, seeds)
    agg = aggregate(eval500, seeds)
    coverage = {g: sum(len(eval500[g].get(str(s), {})) for s in seeds) for g, _ in ARMS}
    total = sum(coverage.values())
    print('ladder coverage (cells with a 500-episode JSON): '
          + ', '.join(f'gamma={g}: {n}' for g, n in coverage.items())
          + f'  |  total {total}')
    report = {'runs': runs,
              'note': ('affine_strict antmaze-medium-navigate, discount sweep 0.98/0.99/'
                       '0.995, 3 seeds each. Ladder cells are 500-episode '
                       'tools/eval_checkpoint.py evals (eval_workers=1); in-loop eval is '
                       '50 episodes (+/-0.115) and OVER-READS on this env; train.csv '
                       'logged every 5k. BC control 0.072; latent-actor arm ~0.21.'),
              'series': series, 'eval500': eval500, 'aggregate': agg}
    with open(os.path.join(out_dir, f'{args.name}.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print(f'wrote {os.path.join(out_dir, args.name)}.json')
    plot(series, eval500, agg, seeds, os.path.join(out_dir, f'{args.name}.png'))
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    write_table(eval500, agg, seeds, os.path.join(repo, 'docs', 'tables', args.table))


if __name__ == '__main__':
    main()
