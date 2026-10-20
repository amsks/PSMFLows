"""Curves over training of what the flowbc actor's bootstrap latent gives the critic, from the
tools/diag_bootstrap_ladder.py reports of two or more tags (2026-10-20).

Reads $PSM_DATA/logs/actor_ladder/<tag>_<epoch>.json for every --tags entry, one panel per
metric group (tools.diag_bootstrap_ladder.METRIC_GROUPS; x = checkpoint step, mean +- SE over
the probe states as error bars, one hue per tag in figstyle's fixed order, one line style per
metric within a panel), writes docs/figures/2026-10-20-actor-ladder/ladder_<tags>.png (and the
numbers next to it as JSON) and prints a table of the deciding metrics per (tag, step).

Run:
  PSM_DATA=... .venv/bin/python tools/fig_bootstrap_ladder.py --tags A_sd0 B_sd0
  (--dir overrides the report directory, --out the figure path, --metrics the table columns)
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle
import matplotlib.pyplot as plt

from tools.diag_bootstrap_ladder import METRIC_GROUPS, ladder_series

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'docs', 'figures', '2026-10-20-actor-ladder')
LINESTYLES = ['-', '--', ':', '-.', (0, (5, 1, 1, 1))]
TABLE_METRICS = ['cos_dbar_gbar', 'cos_dbar_ctop', 'dbar_norm', 'delta_norm', 'dq_actor_online', 'dq_bo2_online',
                 'dq_bo64_online', 'rank_actor_online', 'rank_actor_target', 'nn_actor', 'nn_bc', 'dist_data_actor',
                 'dist_data_bc', 'loss_own', 'loss_bcboot', 'td_target_absmean_own', 'm_softmax_diag_own',
                 'phi_eff_rank']


def load_reports(report_dir, tags):
    """Every <tag>_<epoch>.json under report_dir for the given tags (the tag may itself hold
    underscores; the epoch is the trailing integer). Returns a list of report dicts."""
    reports = []
    for tag in tags:
        pat = re.compile(rf'^{re.escape(tag)}_(\d+)\.json$')
        for path in sorted(glob.glob(os.path.join(report_dir, f'{tag}_*.json'))):
            m = pat.match(os.path.basename(path))
            if not m:
                continue
            with open(path) as f:
                r = json.load(f)
            r['tag'] = tag
            r['restore_epoch'] = int(m.group(1))
            reports.append(r)
        if not any(r['tag'] == tag for r in reports):
            print(f'warning: no reports for tag {tag} under {report_dir}')
    return reports


def draw_panel(ax, series, tags, title, metrics):
    for ti, tag in enumerate(tags):
        color = figstyle.PALETTE[ti % len(figstyle.PALETTE)]
        for mi, m in enumerate(metrics):
            s = series.get(tag, {}).get(m)
            if s is None:
                continue
            x = np.asarray(s['steps'], np.float64)
            ax.errorbar(x, s['mean'], yerr=s['sem'], color=color, linestyle=LINESTYLES[mi % len(LINESTYLES)],
                        marker='o', markersize=3, capsize=2, linewidth=1.2, label=f'{tag}: {m}')
    ax.set_title(title)
    ax.set_xlabel('step')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 1e3:.0f}k'))
    ax.ticklabel_format(axis='y', useOffset=False)
    ax.legend(fontsize=5.5, loc='best')


def make_figure(series, tags, groups, out):
    figstyle.use_style()
    n = len(groups)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 2.6), squeeze=False)
    for ax, (title, metrics) in zip(axes.flat, groups):
        draw_panel(ax, series, tags, title, metrics)
    for ax in list(axes.flat)[n:]:
        ax.axis('off')
    fig.suptitle('bootstrap ladder: ' + ', '.join(tags), y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f'wrote {out}')


def print_table(series, tags, metrics):
    cols = [m for m in metrics if any(m in series.get(t, {}) for t in tags)]
    header = f'{"tag":10s} {"step":>8s} ' + ' '.join(f'{c:>14s}' for c in cols)
    print(header)
    rows = []
    for tag in tags:
        steps = sorted({st for m in series.get(tag, {}).values() for st in m['steps']})
        for st in steps:
            vals = []
            for c in cols:
                s = series[tag].get(c)
                if s is None or st not in s['steps']:
                    vals.append(None)
                else:
                    vals.append(s['mean'][s['steps'].index(st)])
            rows.append((tag, st, vals))
            print(f'{tag:10s} {st:>8d} ' + ' '.join('           n/a' if v is None else f'{v:>14.4g}' for v in vals))
    return {'columns': cols, 'rows': [{'tag': t, 'step': s, 'values': v} for t, s, v in rows]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--dir', default=None, help='report dir (default $PSM_DATA/logs/actor_ladder)')
    ap.add_argument('--out', default=None, help='figure path (default docs/figures/2026-10-20-actor-ladder/ladder_<tags>.png)')
    ap.add_argument('--metrics', nargs='*', default=TABLE_METRICS, help='table columns')
    args = ap.parse_args(argv)
    report_dir = args.dir or os.path.join(os.environ.get('PSM_DATA', '.'), 'logs', 'actor_ladder')
    out = args.out or os.path.join(FIG_DIR, 'ladder_' + '_'.join(args.tags) + '.png')
    reports = load_reports(report_dir, args.tags)
    if not reports:
        sys.exit(f'no reports found under {report_dir} for tags {args.tags}')
    series = ladder_series(reports)
    make_figure(series, args.tags, METRIC_GROUPS, out)
    table = print_table(series, args.tags, args.metrics)
    with open(os.path.splitext(out)[0] + '.json', 'w') as f:
        json.dump({'tags': args.tags, 'report_dir': report_dir, 'series': series, 'table': table,
                   'reports': [r['restore_path'] + ' @ ' + str(r['restore_epoch']) for r in reports]}, f, indent=2)
    print(f'wrote {os.path.splitext(out)[0] + ".json"}')


if __name__ == '__main__':
    main()
