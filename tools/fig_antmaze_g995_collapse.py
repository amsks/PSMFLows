"""Antmaze `discount=0.995` collapse: every logged training scalar, 0.995 vs 0.99.

`docs/design/2026-09-06-antmaze-failure-tests.md` §6.5 item 3 left this object open. The
affine paper-strict agent on antmaze-medium reads 0.296 +/- 0.102 over 50k-250k at
`gamma=0.995` and 0.010 +/- 0.008 over 300k-500k, below the BC control, on all three seeds
at both late checkpoints, while `gamma=0.99` holds 0.252 +/- 0.100 late. This figure asks
what in `train.csv` separates them, and when.

The answer the panels carry: `psm_loss` -- the contrastive TD term -- DIVERGES at
`gamma=0.995` (monotone multiplicative growth, ~one decade per 25k steps, 2.5e3 -> 8e17)
and does not at `gamma=0.99`. It crosses `ortho_coef * |orth_loss|` at ~120k, after which
the orthonormality regulariser is a rounding error against it; `orth_offdiag` then climbs
from its ~64 baseline toward its algebraic MAXIMUM `0.5 * z_dim^2 = 8192`, which is reached
only when every state's `phi` is the same direction. Between 250k and 300k the residual
off-rank-1 energy `1 - orth_offdiag/8192` falls by an order of magnitude on every seed, so
`w = E_D[r phi]` becomes parallel to the one surviving direction and `Q = psi^T w` loses
the state. `psi_q_spread_rel` -- the action-latent signal the inner argmax consumes -- does
NOT move; it is outgrown, not flattened.

`gamma=0.99` seed 1 does the same thing three times later (crosses at 400k, `orth_offdiag`
leaves baseline at 460k) and is the one `gamma=0.99` seed whose 500k cell is 0.004. That is
the within-arm replication: 4 diverging runs collapse, 2 non-diverging runs do not.

Reads only `train.csv` / `eval.csv` under --exp and `eval500_*.json` under --logs; writes
only docs/figures. Idempotent and partial-safe -- missing cells render as gaps.

Usage:
    .venv/bin/python tools/fig_antmaze_g995_collapse.py --logs $PSM_DATA/logs \
        --exp $PSM_DATA/exp/PSMFLows --name 2026-09-07-antmaze-g995-collapse
"""
import argparse
import csv
import glob
import json
import os

import matplotlib

matplotlib.use('Agg')

import figstyle
import matplotlib.pyplot as plt

BC_CONTROL = 0.072            # antmaze, 500 ep, frozen flow acting alone
Z_DIM = 128                   # phi width; ortho_diag is pinned at -Z_DIM by the projection
ORTHO_COEF = 1000.0           # agent.ortho_coef in both arms
ORTH_OFFDIAG_MAX = 0.5 * Z_DIM ** 2      # 8192: reached only when phi is rank one
ORTH_OFFDIAG_BASE = 64.0                 # what a random-ish basis reads

ARMS = {
    'g995': {'group': 'affine_strict_antmaze_g995', 'label': r'$\gamma$=0.995',
             'color': figstyle.PALETTE[1],
             'epochs': [50, 100, 150, 200, 250, 300, 500]},
    'g99': {'group': 'affine_strict_antmaze_g99', 'label': r'$\gamma$=0.99',
            'color': figstyle.PALETTE[0], 'epochs': list(range(50, 550, 50))},
}
SEEDS = [0, 1, 2]
COLLAPSE = (250000, 300000)   # the window the design note asks about

#: (csv column, panel title, log-y) for every scalar train.csv carries, plus two derived.
PANELS = [
    ('training/psm_loss', 'psm_loss  (TD term)', True),
    ('training/psm_offdiag', 'psm_offdiag', True),
    ('training/psm_diag', '|psm_diag|', True),
    ('training/orth_loss', 'orth_loss  (+1000x in the objective)', False),
    ('training/orth_offdiag', r'orth_offdiag  (max 8192 = $\phi$ rank one)', False),
    ('training/orth_diag', 'orth_diag  (pinned at -128)', False),
    ('__resid__', r'$1-$orth_offdiag$/8192$  (off-rank-1 energy)', True),
    ('__qlevel__', '|Q| level  (psi_q_spread / _rel)', True),
    ('training/psi_q_spread', 'psi_q_spread  (absolute)', True),
    ('training/psi_q_spread_rel', 'psi_q_spread_rel  (inner argmax signal)', False),
    ('training/psi_q_index_spread_rel', 'psi_q_index_spread_rel  (GPI signal)', False),
    ('training/psi_q_range_rel', 'psi_q_range_rel', False),
    ('training/w_enc_spread', 'w_enc_spread', False),
    ('__idxratio__', 'index / action spread ratio', True),
    ('time/epoch_time', 'epoch_time (s)', False),
]


def run_dir(exp, group, seed):
    hits = sorted(glob.glob(os.path.join(exp, group, f'sd00{seed}_*')))
    return hits[0] if hits else None


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def series(rows, col):
    """(steps, values) for a train.csv column, including the two derived quantities."""
    steps, vals = [], []
    for r in rows:
        try:
            s = int(float(r['step']))
            if col == '__resid__':
                v = max(1.0 - float(r['training/orth_offdiag']) / ORTH_OFFDIAG_MAX, 1e-8)
            elif col == '__qlevel__':
                v = float(r['training/psi_q_spread']) / max(
                    float(r['training/psi_q_spread_rel']), 1e-12)
            elif col == '__idxratio__':
                v = float(r['training/psi_q_index_spread_rel']) / max(
                    float(r['training/psi_q_spread_rel']), 1e-12)
            elif col == 'training/psm_diag':
                v = abs(float(r['training/psm_diag']))
            else:
                v = float(r[col])
        except (KeyError, ValueError, ZeroDivisionError):
            continue
        steps.append(s)
        vals.append(v)
    return steps, vals


def first_crossing(steps, vals, thresh, above=True):
    for s, v in zip(steps, vals):
        if (v > thresh) if above else (v < thresh):
            return s
    return None


def onsets(rows):
    """The four dated events of the divergence chain, for one run."""
    st, psm = series(rows, 'training/psm_loss')
    _, orth = series(rows, 'training/orth_loss')
    _, off = series(rows, 'training/orth_offdiag')
    _, resid = series(rows, '__resid__')
    cross = None
    for s, p, o in zip(st, psm, orth):
        if p > ORTHO_COEF * abs(o):
            cross = s
            break
    return {
        'psm_gt_1e4': first_crossing(st, psm, 1e4),
        'psm_gt_1e6': first_crossing(st, psm, 1e6),
        'psm_exceeds_ortho_term': cross,
        'orth_offdiag_gt_2x_base': first_crossing(st, off, 2 * ORTH_OFFDIAG_BASE),
        'orth_offdiag_gt_0.99max': first_crossing(st, off, 0.99 * ORTH_OFFDIAG_MAX),
        'orth_offdiag_gt_0.999max': first_crossing(st, off, 0.999 * ORTH_OFFDIAG_MAX),
        'resid_lt_1e-3': first_crossing(st, resid, 1e-3, above=False),
        'psm_loss_final': psm[-1] if psm else None,
        'orth_offdiag_final': off[-1] if off else None,
    }


def eval500(logs, arm, epoch_k, seed):
    p = os.path.join(logs, f'eval500_affine{epoch_k}k_strict_antmaze_{arm}_sd{seed}.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f).get('success')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', default=os.environ.get('PSM_DATA', '') + '/logs')
    ap.add_argument('--exp', default=os.environ.get('PSM_DATA', '') + '/exp/PSMFLows')
    ap.add_argument('--name', default='2026-09-07-antmaze-g995-collapse')
    ap.add_argument('--out-dir', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'figures'))
    args = ap.parse_args()

    data = {}
    for arm, spec in ARMS.items():
        for sd in SEEDS:
            d = run_dir(args.exp, spec['group'], sd)
            if d is None:
                continue
            data[(arm, sd)] = {'dir': d,
                               'train': read_csv(os.path.join(d, 'train.csv')),
                               'eval': read_csv(os.path.join(d, 'eval.csv'))}
    print(f'runs found: {len(data)} / {len(ARMS) * len(SEEDS)}')

    figstyle.use_style()
    ncol, nrow = 4, 5           # 15 scalar panels + eval500 + in-loop success + xy/length
    fig, axes = plt.subplots(nrow, ncol, figsize=(2 * figstyle.TEXTWIDTH, 1.55 * nrow))
    ax = axes.ravel()

    for i, (col, title, logy) in enumerate(PANELS):
        a = ax[i]
        a.axvspan(*[c / 1000 for c in COLLAPSE], color=figstyle.PALETTE[4],
                  alpha=0.12, lw=0)
        for (arm, sd), rec in sorted(data.items()):
            st, v = series(rec['train'], col)
            if not st:
                continue
            a.plot([s / 1000 for s in st], v, color=ARMS[arm]['color'],
                   lw=0.8, alpha=0.85, ls=['-', '--', ':'][sd])
        if logy:
            a.set_yscale('log')
        if col == 'training/orth_offdiag':
            a.axhline(ORTH_OFFDIAG_MAX, color=figstyle.INK_MUTED, lw=0.6, ls='--')
        a.set_title(title, fontsize=7)
        a.tick_params(labelsize=6)

    # --- eval500 ladder
    a = ax[len(PANELS)]
    a.axvspan(*[c / 1000 for c in COLLAPSE], color=figstyle.PALETTE[4], alpha=0.12, lw=0)
    ladder = {}
    for arm, spec in ARMS.items():
        for sd in SEEDS:
            xs, ys = [], []
            for e in spec['epochs']:
                s = eval500(args.logs, arm, e, sd)
                if s is not None:
                    xs.append(e)
                    ys.append(s)
                    ladder[f'{arm}_sd{sd}_{e}k'] = s
            a.plot(xs, ys, color=spec['color'], lw=0.9, alpha=0.85,
                   ls=['-', '--', ':'][sd], marker='o', ms=2)
    a.axhline(BC_CONTROL, color=figstyle.INK_MUTED, lw=0.7, ls='--')
    a.set_title('500-episode success (BC 0.072)', fontsize=7)
    a.tick_params(labelsize=6)

    # --- in-loop success / episode length / visited xy
    for j, (key, title) in enumerate([('evaluation/success', 'in-loop success (50 ep)'),
                                      ('evaluation/episode.length', 'episode length'),
                                      ('evaluation/xy', 'mean visited xy')]):
        a = ax[len(PANELS) + 1 + j]
        a.axvspan(*[c / 1000 for c in COLLAPSE], color=figstyle.PALETTE[4], alpha=0.12, lw=0)
        for (arm, sd), rec in sorted(data.items()):
            xs = [int(float(r['step'])) / 1000 for r in rec['eval']]
            ys = [float(r[key]) for r in rec['eval']]
            a.plot(xs, ys, color=ARMS[arm]['color'], lw=0.9, alpha=0.85,
                   ls=['-', '--', ':'][sd], marker='o', ms=2)
        a.set_title(title, fontsize=7)
        a.tick_params(labelsize=6)

    for a in ax[len(PANELS) + 4:]:
        a.axis('off')
    for a in axes[-1]:
        a.set_xlabel('step (k)', fontsize=7)
    handles = [plt.Line2D([], [], color=s['color'], lw=1.2, label=s['label'])
               for s in ARMS.values()]
    handles += [plt.Line2D([], [], color=figstyle.INK_MUTED, lw=1.0,
                           ls=['-', '--', ':'][s], label=f'seed {s}') for s in SEEDS]
    fig.legend(handles=handles, loc='lower right', ncol=5, frameon=False,
               bbox_to_anchor=(0.98, 0.02))
    fig.suptitle('antmaze-medium affine paper-strict: every logged scalar, '
                 r'$\gamma$=0.995 vs $\gamma$=0.99 (shaded: the 250k$\to$300k collapse)',
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.975))

    os.makedirs(args.out_dir, exist_ok=True)
    png = os.path.join(args.out_dir, args.name + '.png')
    fig.savefig(png)
    print(f'figure -> {png}')

    report = {
        'note': ('affine paper-strict antmaze-medium, gamma=0.995 vs 0.99, three seeds '
                 'each. orth_offdiag maxes at 0.5*z_dim^2 = 8192 (phi rank one); '
                 'psm_loss is the contrastive TD term and the objective is '
                 'psm_loss + 1000*orth_loss.'),
        'runs': {f'{a}_sd{s}': r['dir'] for (a, s), r in sorted(data.items())},
        'constants': {'z_dim': Z_DIM, 'ortho_coef': ORTHO_COEF,
                      'orth_offdiag_max': ORTH_OFFDIAG_MAX,
                      'orth_offdiag_baseline': ORTH_OFFDIAG_BASE,
                      'bc_control': BC_CONTROL},
        'onsets': {f'{a}_sd{s}': onsets(r['train']) for (a, s), r in sorted(data.items())},
        'eval500': ladder,
        'series': {},
    }
    for (a, s), rec in sorted(data.items()):
        entry = {}
        for col, _, _ in PANELS + [('__none__', '', False)]:
            if col == '__none__':
                continue
            st, v = series(rec['train'], col)
            entry['step'] = st
            entry[col.strip('_').replace('training/', '')] = v
        entry['eval_step'] = [int(float(r['step'])) for r in rec['eval']]
        for k in ('evaluation/success', 'evaluation/episode.length', 'evaluation/xy',
                  'evaluation/episode.return'):
            entry[k.split('/')[-1]] = [float(r[k]) for r in rec['eval']]
        report['series'][f'{a}_sd{s}'] = entry

    jpath = os.path.join(args.out_dir, args.name + '.json')
    with open(jpath, 'w') as f:
        json.dump(report, f, indent=1)
    print(f'json -> {jpath}')
    for k, v in report['onsets'].items():
        print(f'  {k}: psm>1e4 @{v["psm_gt_1e4"]}  crosses ortho @'
              f'{v["psm_exceeds_ortho_term"]}  phi departs @{v["orth_offdiag_gt_2x_base"]}'
              f'  resid<1e-3 @{v["resid_lt_1e-3"]}')


if __name__ == '__main__':
    main()
