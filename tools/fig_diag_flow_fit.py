"""Figures for the behaviour-flow FIT-vs-MEMORIZE diagnostic (tools/diag_flow_fit.py).

Reads every `diag_flow_fit_<env>.json` in the log dir and draws three figures:

  fig_flow_fit_recall        (a) all five curves against the epsilon sweep, one panel per
                                 env. Solid = exact ODE-100 decode, dashed = the distilled
                                 one-step head. `recall_own` is epsilon-free, so it is a
                                 horizontal reference, not a curve.
  fig_flow_fit_memorization  (b) the train-vs-val split of the epsilon-free per-anchor
                                 min-MSE to the anchor's OWN action -- the memorization
                                 line -- with the point-preimage ceiling and the
                                 data-matches-itself value as the two scales that say
                                 whether a number is small.
  fig_flow_fit_min_vs_n      (c) recall_ball at eps = 1x against the sample budget N. A
                                 min over N draws falls with N by construction; this is the
                                 curve that says how much of a reported number is the model.

NOTE on (b): the diagnostic persists summaries (mean, median, p90, bootstrap CI), not the
per-anchor values, so the "distribution" here is the quantile sketch those summaries
support -- median-to-p90 span, mean with its CI -- and not a histogram. Nothing is
interpolated between the stored quantiles.

Not to be confused with tools/fig_flow_fit.py, which is paper figure F2 (flow fidelity vs
a random-flow control, inversion typicality, round-trip error).

Run: PSM_DATA=... .venv/bin/python tools/fig_diag_flow_fit.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# figstyle sets the Agg backend, so it must be imported before pyplot.
from tools.figstyle import FIG_DIR, GRID, INK, INK_MUTED, PALETTE, use_style  # isort: skip

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: figstyle.LOG_DIR points at the midi-01 scratch; on any other machine PSM_DATA wins.
LOGS = (os.environ.get('FLOW_FIT_LOGS')
        or os.path.join(os.environ.get('PSM_DATA', '/mnt/home/amohan/psm-data'), 'logs'))
TABLE_OUT = os.path.join(REPO, 'docs', 'tables', 'flow_fit.md')
#: display order; anything else found is appended alphabetically.
ENV_ORDER = ['pointmaze', 'cube', 'antmaze']
#: epsilon multiples quoted in the table.
TABLE_EPS = (0.5, 1.0, 2.0, 3.0)
#: val anchors are not in the k-NN index, so their ball is drawn entirely from training
#: rows; below this radius most val balls are empty and the statistics are noise.
VAL_VALID_EPS = 1.0

C_OWN_TR, C_OWN_VA = PALETTE[0], PALETTE[5]
C_BALL_TR, C_BALL_VA = PALETTE[1], PALETTE[2]
C_SELF, C_NULL = PALETTE[3], PALETTE[4]


def load_reports(logs=LOGS):
    """{env_key: report} for every diag_flow_fit_<env>.json, smoke runs excluded."""
    out = {}
    for path in sorted(glob.glob(os.path.join(logs, 'diag_flow_fit_*.json'))):
        key = os.path.basename(path)[len('diag_flow_fit_'):-len('.json')]
        if 'smoke' in key:
            continue
        with open(path) as f:
            out[key] = json.load(f)
        out[key]['_path'] = path
    order = [e for e in ENV_ORDER if e in out] + sorted(k for k in out if k not in ENV_ORDER)
    return {k: out[k] for k in order}


def g(summary, field='mean'):
    """A summary dict with n == 0 carries no fields; every read goes through here."""
    if not isinstance(summary, dict):
        return float('nan')
    v = summary.get(field, float('nan'))
    return float('nan') if v is None else float(v)


def curve(report, split, mode, stat, field='mean'):
    """`stat` across the epsilon grid for one (split, decode) panel."""
    per = report['panels'][f'{split}_{mode}']['per_eps']
    return np.array([g(p[stat], field) for p in per], float)


def eps_multiples(report):
    return np.asarray(report['eps_grid']['multiples'], float)


def _finite(x, y):
    m = np.isfinite(y)
    return np.asarray(x)[m], np.asarray(y)[m]


def _plot(ax, x, y, **kw):
    """Draw only the finite part -- data_self at 0.25x has no ball with >= 2 members."""
    xf, yf = _finite(x, y)
    if xf.size:
        ax.plot(xf, yf, **kw)
    return xf.size


# --------------------------------------------------------------------------- (a) recall
def fig_recall(reports):
    n = len(reports)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 2.9), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (env, rep) in zip(axes, reports.items()):
        x = eps_multiples(rep)
        # the region where a val anchor's ball is usually empty
        ax.axvspan(x.min() - 0.05, VAL_VALID_EPS, color=GRID, alpha=0.45, lw=0, zorder=0)
        ax.text(x.min(), 0.985, ' val balls mostly empty', fontsize=6, color=INK_MUTED,
                va='top', ha='left', transform=ax.get_xaxis_transform(), zorder=1)

        for mode, ls in (('ode', '-'), ('onestep', '--')):
            # epsilon-free: a horizontal reference, drawn across the sweep.
            # train and val sit within ~1% of each other, so the wider train line shows
            # as a halo under the val line rather than being hidden by it.
            for split, col, lw in (('train', C_OWN_TR, 2.6), ('val', C_OWN_VA, 1.0)):
                v = g(rep['panels'][f'{split}_{mode}']['recall_own'])
                ax.plot(x, np.full_like(x, v), ls=ls, color=col, lw=lw, alpha=0.8)
            # train and val recall_ball land nearly on top of each other -- markers, not
            # colour alone, are what separates them where they coincide.
            _plot(ax, x, curve(rep, 'train', mode, 'recall_ball'), ls=ls, color=C_BALL_TR,
                  marker='o', ms=3)
            xv, yv = x, curve(rep, 'val', mode, 'recall_ball')
            keep = xv >= VAL_VALID_EPS
            _plot(ax, xv[keep], yv[keep], ls=ls, color=C_BALL_VA, marker='s', ms=3,
                  mfc='none')
            _plot(ax, x, curve(rep, 'train', mode, 'null'), ls=ls, color=C_NULL, lw=1.1)
        # decoder-free: identical in both panels, drawn once.
        _plot(ax, x, curve(rep, 'train', 'ode', 'data_self'), ls='-', color=C_SELF, lw=1.6)

        sat = [p['saturated_frac'] for p in rep['panels']['train_ode']['per_eps']]
        hit = [xx for xx, ss in zip(x, sat) if ss > 0.5]
        if hit:
            ax.axvline(hit[0], color=INK_MUTED, ls=(0, (1, 2)), lw=1.0)
            ax.annotate(f"ball hits k_max={rep['k_max']} ", xy=(hit[0], 0.02),
                        xycoords=ax.get_xaxis_transform(), fontsize=6, color=INK_MUTED,
                        ha='right', va='bottom')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\epsilon$ / median 1-NN state distance')
        ax.set_title(f"{env}  ({rep['panels']['train_ode']['recall_own']['n']} anchors, "
                     f"N={rep['n_samples_per_anchor']})", color=INK)
        ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        ax.set_xlim(x.min() - 0.05, x.max() + 0.05)
    axes[0].set_ylabel(r'distance / mean$|a|$  (log)')

    handles = [plt.Line2D([], [], color=c, marker=mk, ms=3, mfc=mfc, label=l)
               for c, mk, mfc, l in (
        (C_OWN_TR, '', 'none', 'recall_own, train ($\\epsilon$-free)'),
        (C_OWN_VA, '', 'none', 'recall_own, val ($\\epsilon$-free)'),
        (C_BALL_TR, 'o', C_BALL_TR, 'recall_ball, train'),
        (C_BALL_VA, 's', 'none', r'recall_ball, val ($\epsilon \geq 1\times$)'),
        (C_SELF, '', 'none', 'data_self (data vs itself)'),
        (C_NULL, '', 'none', 'null (ball moved, size held)'))]
    handles += [plt.Line2D([], [], color=INK_MUTED, ls='-', label='exact ODE-100 decode'),
                plt.Line2D([], [], color=INK_MUTED, ls='--', label='one-step distilled head')]
    fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.22))
    fig.suptitle('Does the frozen behaviour flow cover the local action cloud?',
                 y=1.04, fontsize=9, color=INK)
    return fig


# --------------------------------------------------------------- (b) memorization split
def fig_memorization(reports):
    n = len(reports)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 2.6), squeeze=False)
    axes = axes[0]
    for ax, (env, rep) in zip(axes, reports.items()):
        rows = rep['min_mse']['ode']
        ymap = {'train': 1.0, 'val': 0.0}
        for split, col in (('train', C_OWN_TR), ('val', C_OWN_VA)):
            own = rows[split]['own']
            y = ymap[split]
            med, p90, mean = g(own, 'median'), g(own, 'p90'), g(own)
            ci = own.get('ci95') or [mean, mean]
            ax.plot([med, p90], [y, y], color=col, lw=2.6, solid_capstyle='round', alpha=0.55)
            ax.plot([med], [y], 'o', color=col, ms=5)
            ax.plot([p90], [y], '|', color=col, ms=9, mew=1.4)
            ax.plot([ci[0], ci[1]], [y, y], color=col, lw=1.0)
            ax.plot([mean], [y], 'D', color=col, ms=4.5)
            ax.text(med, y + 0.16, split, fontsize=7, color=col, ha='center')

        ds = g(rows['train']['data_self_at_1x'])
        if np.isfinite(ds) and ds > 0:
            ax.axvline(ds, color=INK_MUTED, ls='-.', lw=1.0)
            ax.annotate('data vs itself ', xy=(ds, 1.68), fontsize=6, color=INK_MUTED,
                        ha='right', va='top')
        # the exact-inverse ceiling sits 5-6 decades lower; as a vline it would flatten
        # the axis, so it is quoted instead.
        ceil = g(rep.get('point_preimage_ceiling') or {})
        ax.text(0.02, 0.03,
                'point-preimage ceiling: '
                + ('--' if not np.isfinite(ceil) else f'{ceil ** 2:.1e}'),
                fontsize=6, color=INK_MUTED, ha='left', va='bottom',
                transform=ax.transAxes)

        ratio = rows.get('val_over_train_own_mean', float('nan'))
        ax.text(0.02, 0.14, f'val/train mean = {ratio:.3f}', fontsize=7, color=INK,
                ha='left', va='bottom', transform=ax.transAxes)
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=20))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1),
                                              numticks=40))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(-0.55, 1.75)
        ax.set_yticks([])
        ax.set_xlabel(r'min-MSE to own action  $/\ (\mathrm{mean}|a|)^2$')
        ax.set_title(env, color=INK)
        ax.grid(axis='y', visible=False)
    handles = [plt.Line2D([], [], color=INK_MUTED, marker='o', ls='', label='median'),
               plt.Line2D([], [], color=INK_MUTED, marker='D', ls='', label='mean (line = 95% CI)'),
               plt.Line2D([], [], color=INK_MUTED, marker='|', ls='', mew=1.4, label='p90')]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.16))
    fig.suptitle('Memorization: can the flow produce a HELD-OUT state\'s own action '
                 'as well as a training one? (ODE-100)', y=1.05, fontsize=9, color=INK)
    return fig


# ------------------------------------------------------------------------- (c) min_vs_n
def fig_min_vs_n(reports):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for i, (env, rep) in enumerate(reports.items()):
        mvn = rep.get('min_vs_n') or {}
        col = PALETTE[i % len(PALETTE)]
        for key, ls, tag in (('curve', '-', 'ODE-100'), ('curve_onestep', '--', 'one-step')):
            cur = mvn.get(key)
            if not cur:
                continue
            xs = np.array([c['n_samples'] for c in cur], float)
            ys = np.array([g(c['recall_ball']) for c in cur], float)
            lo = np.array([(c['recall_ball'].get('ci95') or [np.nan, np.nan])[0] for c in cur])
            hi = np.array([(c['recall_ball'].get('ci95') or [np.nan, np.nan])[1] for c in cur])
            ax.plot(xs, ys, ls=ls, color=col, marker='o', ms=3, label=f'{env} ({tag})')
            ax.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0)
        ds = g(rep['panels']['train_ode']['per_eps'][
            list(eps_multiples(rep)).index(1.0)]['data_self'])
        if np.isfinite(ds):
            ax.axhline(ds, color=col, ls=':', lw=0.9)
            ax.text(1.05, ds, f'{env} data_self', fontsize=6, color=col, va='bottom')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('sample budget N per anchor')
    ax.set_ylabel(r'recall_ball at $\epsilon=1\times$  (distance / mean$|a|$)')
    ax.set_title('A min over $N$ draws falls with $N$ by construction\n'
                 '(the diagnostic records this curve for train anchors, ODE decode only)',
                 color=INK, fontsize=8)
    ax.legend(loc='upper right')
    return fig


# ------------------------------------------------------------------------------ (d) table
def _fmt(v, p=4):
    return '--' if v is None or not np.isfinite(v) else f'{v:.{p}f}'


def build_table(reports):
    hdr = ('| env | decoder | own mMSE train | own mMSE val | val/train | ball mMSE @1x | '
           'data_self mMSE @1x | own/data_self | recall_ball @0.5x | @1x | @2x | @3x | '
           'null @1x | ball sat @3x |')
    sep = '|' + '---|' * 14
    lines = [
        '# Behaviour-flow fit vs memorization (`tools/diag_flow_fit.py`)',
        '',
        'Generated by `tools/fig_diag_flow_fit.py` -- do not hand-edit.',
        '',
        'mMSE = per-anchor minimum squared distance over the flow sample cloud, in units of',
        '`(mean|a|)^2`; recall_ball / null / data_self are distances in units of `mean|a|`.',
        'Latents are unclipped `N(0, I)`; the ball is always built from TRAINING rows.',
        '"own" is the epsilon-free line: the anchor\'s OWN recorded action.',
        '"ball sat @3x" is the fraction of anchors whose 3x ball hits the k_max neighbour',
        'cap -- where it is large, recall_ball at that radius is a truncated ball.',
        '',
        hdr, sep,
    ]
    per_env = {}
    for env, rep in reports.items():
        mults = list(eps_multiples(rep))
        j1 = mults.index(1.0)
        per_env[env] = {}
        for mode in ('ode', 'onestep'):
            rows = rep['min_mse'][mode]
            rb = curve(rep, 'train', mode, 'recall_ball')
            rec = {
                'own_train': g(rows['train']['own']), 'own_val': g(rows['val']['own']),
                'own_train_median': g(rows['train']['own'], 'median'),
                'own_val_median': g(rows['val']['own'], 'median'),
                'own_train_p90': g(rows['train']['own'], 'p90'),
                'own_val_p90': g(rows['val']['own'], 'p90'),
                'val_over_train': rows.get('val_over_train_own_mean', float('nan')),
                'ball_1x': g(rows['train']['ball_at_1x']),
                'data_self_1x': g(rows['train']['data_self_at_1x']),
                'own_over_data_self': rows['train'].get('ratio_own_over_data_self',
                                                        float('nan')),
                'recall_ball': {m: (rb[mults.index(m)] if m in mults else float('nan'))
                                for m in TABLE_EPS},
                'null_1x': g(rep['panels'][f'train_{mode}']['per_eps'][j1]['null']),
                'ball_saturated_at_3x': float(
                    rep['panels'][f'train_{mode}']['per_eps'][mults.index(3.0)][
                        'saturated_frac']) if 3.0 in mults else float('nan'),
            }
            per_env[env][mode] = rec
            lines.append('| ' + ' | '.join([
                env, mode, _fmt(rec['own_train']), _fmt(rec['own_val']),
                _fmt(rec['val_over_train'], 3), _fmt(rec['ball_1x']),
                _fmt(rec['data_self_1x']), _fmt(rec['own_over_data_self'], 3),
                *[_fmt(rec['recall_ball'][m], 3) for m in TABLE_EPS],
                _fmt(rec['null_1x'], 3), _fmt(rec['ball_saturated_at_3x'], 2)]) + ' |')
    prov_hdr = ('| env | report | flow | epoch | anchors | N | train rows | val rows | '
                'median 1-NN dist | point-preimage ceiling (dist) |')
    lines += ['', '## Provenance', '', prov_hdr, '|' + '---|' * 10]
    for env, rep in reports.items():
        ceil = rep.get('point_preimage_ceiling') or {}
        lines.append('| ' + ' | '.join([
            env, '`' + os.path.basename(rep['_path']) + '`',
            '`' + os.path.basename(str(rep['restore_path'])) + '`', str(rep['restore_epoch']),
            str(rep['n_anchors']), str(rep['n_samples_per_anchor']),
            str(rep['train_rows']), str(rep['val_rows']),
            _fmt(rep['eps_grid']['median_1nn_dist'], 3),
            _fmt(g(ceil), 5) if ceil else '-- (no preimage npz)']) + ' |')
    return '\n'.join(lines) + '\n', per_env


def save_fig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    paths = []
    for ext in ('pdf', 'png'):
        p = os.path.join(FIG_DIR, f'{name}.{ext}')
        fig.savefig(p)
        paths.append(p)
    for p in paths:
        print(f'wrote {p}')
    return paths


def main():
    use_style()
    reports = load_reports()
    if not reports:
        raise SystemExit(f'no diag_flow_fit_*.json under {LOGS}')
    print(f'envs: {", ".join(reports)}  (from {LOGS})')

    save_fig(fig_recall(reports), 'fig_flow_fit_recall')
    save_fig(fig_memorization(reports), 'fig_flow_fit_memorization')
    save_fig(fig_min_vs_n(reports), 'fig_flow_fit_min_vs_n')

    table, per_env = build_table(reports)
    print()
    print(table)
    os.makedirs(os.path.dirname(TABLE_OUT), exist_ok=True)
    with open(TABLE_OUT, 'w') as f:
        f.write(table)
    print(f'wrote {TABLE_OUT}')
    # every number in a figure also lives in a JSON report (figstyle.save's contract).
    data_dir = os.path.join(FIG_DIR, 'data')
    os.makedirs(data_dir, exist_ok=True)
    for path in (os.path.join(data_dir, 'fig_flow_fit.json'),
                 os.path.join(LOGS, 'fig_flow_fit.json')):
        with open(path, 'w') as f:
            json.dump({'sources': {e: r['_path'] for e, r in reports.items()},
                       'table': per_env}, f, indent=2)
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
