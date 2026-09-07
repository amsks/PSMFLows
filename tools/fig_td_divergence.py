"""Plot training diagnostics from run directories to see the measure TD loss explode.

Reads train.csv / eval.csv directly off disk under
`<exp>/<group>/sdNNN_*/{train,eval}.csv` -- no wandb, no pandas dependency (csv + numpy
only, since pandas may not be installed in .venv). Produces four PNGs:

  2026-09-07-td-divergence-discount.png   -- antmaze, gamma in {0.98, 0.99, 0.995}
  2026-09-07-td-divergence-pessimism.png  -- antmaze gamma=0.995, kappa in {0, .25, .5, 1}
  2026-09-07-td-divergence-levers.png     -- antmaze gamma=0.995, ortho/psi-bound/tau/free-psi levers
  2026-09-07-td-divergence-mpess0-live.png -- kappa=0.5 vs in-flight kappa=0 fix, antmaze+cube

and the numbers plotted, as JSON, to docs/figures/2026-09-07-td-divergence-curves.json.

Usage:
    .venv/bin/python tools/fig_td_divergence.py
    .venv/bin/python tools/fig_td_divergence.py --exp /other/exp/root --out_dir /tmp/figs
"""
import argparse
import csv
import glob
import json
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DEFAULT_EXP = '/mnt/home/amohan/psm-data/exp/PSMFLows'

# Fixed categorical order -- never cycle past six.
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
INK = '#3a3a3a'        # dark grey, general text
GRID = '#dcdcdc'       # recessive grid

SEED_DIR_RE = re.compile(r'^sd(\d+)_')


def use_style():
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.facecolor': 'white',
        'font.size': 9,
        'axes.titlesize': 9.5,
        'axes.labelsize': 8.5,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'legend.fontsize': 8,
        'axes.edgecolor': INK,
        'axes.labelcolor': INK,
        'text.color': INK,
        'xtick.color': INK,
        'ytick.color': INK,
        'axes.linewidth': 0.7,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': GRID,
        'grid.linewidth': 0.6,
        'legend.frameon': False,
    })


def find_seed_runs(exp_dir, group):
    """{seed_int: run_dir} for a group, sorted by seed. Skips groups that don't exist."""
    group_dir = os.path.join(exp_dir, group)
    out = {}
    if not os.path.isdir(group_dir):
        return out
    for hit in sorted(glob.glob(os.path.join(group_dir, 'sd*_*'))):
        base = os.path.basename(hit)
        m = SEED_DIR_RE.match(base)
        if not m:
            continue
        out[int(m.group(1))] = hit
    return out


def read_csv_cols(path):
    """{column_name: np.array(float)}; '' / 'nan' -> np.nan. Missing file -> {}."""
    if not os.path.exists(path):
        return {}
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    cols = rows[0].keys()
    out = {}
    for c in cols:
        vals = []
        for r in rows:
            v = r.get(c, '')
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(np.nan)
        out[c] = np.array(vals, dtype=float)
    return out


def load_run(run_dir):
    return {'train': read_csv_cols(os.path.join(run_dir, 'train.csv')),
            'eval': read_csv_cols(os.path.join(run_dir, 'eval.csv'))}


def load_group_series(exp_dir, group):
    """{seed_int: {'train': {...}, 'eval': {...}}} for every sdNNN dir found."""
    return {seed: load_run(run_dir) for seed, run_dir in find_seed_runs(exp_dir, group).items()}


def masked_for_log(y):
    y = np.array(y, dtype=float)
    y = np.where(y > 0, y, np.nan)
    return y


def plot_panel(ax, group_series, source, metric, yscale, xlim, color, label,
                json_out, series_key):
    """Draw every seed's line for one series on one panel; record what was plotted."""
    plotted_any = False
    last_seed0 = None
    seeds = sorted(group_series.keys())
    for seed in seeds:
        block = group_series[seed].get(source, {})
        step = block.get('step')
        y = block.get(metric)
        if step is None or y is None:
            continue  # column/source absent for this run -- skip this panel for this seed
        step = np.asarray(step, dtype=float)
        y = np.asarray(y, dtype=float)
        n = min(len(step), len(y))
        step, y = step[:n], y[:n]
        if xlim is not None:
            keep = step <= xlim[1]
            step, y = step[keep], y[keep]
        if len(step) == 0:
            continue
        yplot = masked_for_log(y) if yscale == 'log' else y
        alpha = 1.0 if seed == seeds[0] else 0.55
        ax.plot(step, yplot, color=color, lw=1.4, alpha=alpha)
        plotted_any = True

        json_out.setdefault(series_key, {}).setdefault(str(seed), {})[metric] = [
            [float(s), (None if (np.isnan(v)) else float(v))] for s, v in zip(step, y)]

        if seed == seeds[0]:
            finite = np.where(np.isfinite(yplot))[0]
            if len(finite):
                last_seed0 = (step[finite[-1]], yplot[finite[-1]])

    ax.set_yscale(yscale)
    ax.grid(True, color=GRID, linewidth=0.6)
    if xlim is not None:
        ax.set_xlim(*xlim)
    return plotted_any, last_seed0


def dodge_labels(ax, items, min_sep_px=15):
    """items: [(x_data, y_data, color, label), ...] -> same, with y nudged in *display*
    space (so it works correctly under log-scaled axes) so labels don't overlap."""
    if not items:
        return items
    pts = [ax.transData.transform((x, y)) for x, y, _, _ in items]
    order = sorted(range(len(items)), key=lambda i: pts[i][1])
    pys = [pts[i][1] for i in order]
    for i in range(1, len(pys)):
        if pys[i] - pys[i - 1] < min_sep_px:
            pys[i] = pys[i - 1] + min_sep_px
    bbox = ax.get_window_extent()
    if pys[-1] > bbox.y1:
        shift = pys[-1] - bbox.y1
        pys = [p - shift for p in pys]
    if pys[0] < bbox.y0:
        shift = bbox.y0 - pys[0]
        pys = [p + shift for p in pys]
    result = list(items)
    for k, i in enumerate(order):
        x, y, color, label = items[i]
        px = pts[i][0]
        new_x, new_y = ax.transData.inverted().transform((px, pys[k]))
        result[i] = (new_x, new_y, color, label)
    return result


def draw_end_labels(ax, end_labels):
    for x, y, color, label in dodge_labels(ax, end_labels):
        ax.annotate(label, xy=(x, y), xycoords='data',
                    xytext=(6, 0), textcoords='offset points',
                    va='center', ha='left', fontsize=6.5, color=color,
                    annotation_clip=False)


def build_grid(fig_title, panels, series_defs, exp_dir, xlim, out_path, json_root,
                figsize=(15, 8)):
    """panels: [(source, metric, title, yscale), ...] laid out on a 2x3 grid.
    series_defs: [(label, group, color), ...].
    """
    use_style()
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.ravel()

    group_cache = {group: load_group_series(exp_dir, group) for _, group, _ in series_defs}
    fig_json = {}
    label_positions = []  # collision-avoidance bookkeeping, per axis

    for ax, (source, metric, title, yscale) in zip(axes, panels):
        end_labels = []
        for label, group, color in series_defs:
            gs = group_cache[group]
            if not gs:
                continue
            plotted, last_seed0 = plot_panel(ax, gs, source, metric, yscale, xlim, color,
                                              label, fig_json, label)
            if plotted and last_seed0 is not None:
                end_labels.append((last_seed0[0], last_seed0[1], color, label))
        ax.set_title(title)
        ax.set_xlabel('step')
        if end_labels:
            draw_end_labels(ax, end_labels)
        else:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center', va='center',
                    color=INK, fontsize=8)

    handles = [Line2D([0], [0], color=color, lw=2, label=label)
               for label, _, color in series_defs]
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 6),
               bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.suptitle(fig_title, color=INK, fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 0.93, 0.95])
    fig.savefig(out_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    json_root[os.path.basename(out_path)] = fig_json
    print(f'wrote {out_path}')


PANELS_6 = [
    ('train', 'training/psm_loss', 'psm_loss', 'log'),
    ('train', 'training/psm_offdiag', 'psm_offdiag', 'log'),
    ('train', 'training/psm_diag', 'psm_diag', 'linear'),
    ('train', 'training/orth_offdiag', 'orth_offdiag', 'log'),
    ('train', 'training/psi_q_index_spread_rel', 'psi_q_index_spread_rel', 'linear'),
    ('eval', 'evaluation/success', 'success (in-loop, 50 ep)', 'linear'),
]


def fig_discount(exp_dir, out_dir, json_root):
    series_defs = [
        (r'$\gamma$=0.98', 'affine_strict_antmaze', COLORS[0]),
        (r'$\gamma$=0.99', 'affine_strict_antmaze_g99', COLORS[1]),
        (r'$\gamma$=0.995', 'affine_strict_antmaze_g995', COLORS[2]),
    ]
    out_path = os.path.join(out_dir, '2026-09-07-td-divergence-discount.png')
    build_grid('antmaze-medium-navigate: varying discount gamma (unfixed agent, kappa=0.5)',
               PANELS_6, series_defs, exp_dir, xlim=(0, 500000), out_path=out_path,
               json_root=json_root)


def fig_pessimism(exp_dir, out_dir, json_root):
    series_defs = [
        (r'$\kappa$=0', 'mla_g995_pess0', COLORS[0]),
        (r'$\kappa$=0.25', 'mla_g995_pess025', COLORS[1]),
        (r'$\kappa$=0.5 (unfixed)', 'affine_strict_antmaze_g995', COLORS[2]),
        (r'$\kappa$=1.0', 'mla_g995_pess10', COLORS[3]),
    ]
    out_path = os.path.join(out_dir, '2026-09-07-td-divergence-pessimism.png')
    build_grid('antmaze-medium-navigate, gamma=0.995: pessimism dose kappa (150k steps)',
               PANELS_6, series_defs, exp_dir, xlim=(0, 150000), out_path=out_path,
               json_root=json_root)


def fig_levers(exp_dir, out_dir, json_root):
    series_defs = [
        ('unfixed', 'affine_strict_antmaze_g995', COLORS[0]),
        ('relative ortho', 'affine_strict_antmaze_g995_orel', COLORS[1]),
        ('psi bound (tanh)', 'affine_strict_antmaze_g995_pbound', COLORS[2]),
        ('ortho 1e5', 'mla_g995_oc1e5', COLORS[3]),
        ('tau 1e-3', 'mla_g995_tau1e3', COLORS[4]),
        ('free psi', 'mla_g995_freepsi', COLORS[5]),
    ]
    out_path = os.path.join(out_dir, '2026-09-07-td-divergence-levers.png')
    build_grid('antmaze-medium-navigate, gamma=0.995: other stabilization levers',
               PANELS_6, series_defs, exp_dir, xlim=(0, 500000), out_path=out_path,
               json_root=json_root)


PANELS_3 = [
    ('train', 'training/psm_loss', 'psm_loss', 'log'),
    ('train', 'training/orth_offdiag', 'orth_offdiag', 'log'),
    ('eval', 'evaluation/success', 'success (in-loop, 50 ep)', 'linear'),
]


def fig_mpess0_live(exp_dir, out_dir, json_root):
    use_style()
    rows = [
        ('antmaze-medium-navigate (gamma=0.99)', 'affine_strict_antmaze_g99',
         'affine_strict_antmaze_g99_mpess0'),
        ('cube-single-play (gamma=0.98)', 'affine_strict_cube', 'affine_strict_cube_mpess0'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    fig_json = {}
    for row_i, (row_title, unfixed_group, mpess0_group) in enumerate(rows):
        series_defs = [
            (r'$\kappa$=0.5 (unfixed)', unfixed_group, COLORS[0]),
            (r'$\kappa$=0 (in flight, partial)', mpess0_group, COLORS[1]),
        ]
        group_cache = {g: load_group_series(exp_dir, g) for _, g, _ in series_defs}
        for col_i, (source, metric, title, yscale) in enumerate(PANELS_3):
            ax = axes[row_i, col_i]
            end_labels = []
            for label, group, color in series_defs:
                gs = group_cache[group]
                if not gs:
                    continue
                plotted, last_seed0 = plot_panel(ax, gs, source, metric, yscale, (0, 500000),
                                                  color, label, fig_json, label)
                if plotted and last_seed0 is not None:
                    end_labels.append((last_seed0[0], last_seed0[1], color, label))
            ax.set_title(title)
            ax.set_xlabel('step')
            if col_i == 0:
                ax.set_ylabel(row_title, fontsize=8, color=INK)
            if end_labels:
                draw_end_labels(ax, end_labels)
            else:
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center', va='center',
                        color=INK, fontsize=8)

    handles = [Line2D([0], [0], color=COLORS[0], lw=2, label=r'$\kappa$=0.5 (unfixed)'),
               Line2D([0], [0], color=COLORS[1], lw=2, label=r'$\kappa$=0 (in flight, partial)')]
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05),
               frameon=False)
    fig.suptitle('mean-pessimism fix in flight: kappa=0.5 (unfixed) vs kappa=0 (partial runs)',
                 color=INK, fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 0.95, 0.94])
    out_path = os.path.join(out_dir, '2026-09-07-td-divergence-mpess0-live.png')
    fig.savefig(out_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    json_root[os.path.basename(out_path)] = fig_json
    print(f'wrote {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default=DEFAULT_EXP, help='experiment root holding run groups')
    ap.add_argument('--out_dir', default=None, help='default: docs/figures')
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out_dir or os.path.join(repo, 'docs', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    json_root = {}
    fig_discount(args.exp, out_dir, json_root)
    fig_pessimism(args.exp, out_dir, json_root)
    fig_levers(args.exp, out_dir, json_root)
    fig_mpess0_live(args.exp, out_dir, json_root)

    json_path = os.path.join(out_dir, '2026-09-07-td-divergence-curves.json')
    with open(json_path, 'w') as f:
        json.dump(json_root, f)
    print(f'wrote {json_path}')


if __name__ == '__main__':
    main()
