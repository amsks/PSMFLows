"""Shared matplotlib style for the paper figures (tools/fig_*.py).

One style module so every figure in the paper reads as one system. The categorical
palette is Okabe-Ito in a fixed order, validated colorblind-safe: adjacent-pair CVD
separation passes (worst 9.6 deutan), normal-vision separation passes (worst 20.0).
Assign hues in this order and never cycle -- a 7th series folds into "other" or becomes
a small multiple instead.

Several palette entries sit below 3:1 contrast against white, so every series carries a
legend entry or a direct label; identity is never color-alone.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#: Fixed categorical order. Blue, vermillion, green, orange, purple, sky.
PALETTE = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7', '#56B4E9']

#: Single-hue sequential ramp (light -> dark) for magnitude.
SEQUENTIAL = 'Blues'

INK = '#1a1a1a'        # primary text
INK_MUTED = '#6b6b6b'  # secondary text, axis labels
GRID = '#d9d9d9'       # recessive grid

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'PAPER', 'ICLR', 'figures')
LOG_DIR = '/data-local/amsks/PSMFLows/logs'

#: \textwidth of the ICLR style, in inches.
TEXTWIDTH = 5.5


def use_style():
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'font.size': 8,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.prop_cycle': plt.cycler(color=PALETTE),
        'axes.edgecolor': INK_MUTED,
        'axes.labelcolor': INK,
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': GRID,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.8,
        'xtick.color': INK_MUTED,
        'ytick.color': INK_MUTED,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'lines.linewidth': 1.4,
        'lines.markersize': 4,
        'legend.frameon': False,
        'text.color': INK,
    })


def save(fig, name, report=None):
    """Write PDF (vector, for the paper) + PNG (preview) + the numbers as JSON.

    No figure in the paper is allowed to carry a number that is not also in a JSON
    report -- hand-copied numbers are how results drift from what was actually run.
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    pdf = os.path.join(FIG_DIR, f'{name}.pdf')
    fig.savefig(pdf)
    fig.savefig(os.path.join(FIG_DIR, f'{name}.png'))
    out = [pdf]
    if report is not None:
        # Next to the figure (so the paper build is reproducible off-machine) and in the
        # log dir (so it sits with every other report).
        data_dir = os.path.join(FIG_DIR, 'data')
        os.makedirs(data_dir, exist_ok=True)
        for path in (os.path.join(data_dir, f'{name}.json'),
                     os.path.join(LOG_DIR, f'{name}.json')):
            with open(path, 'w') as f:
                json.dump(report, f, indent=2)
            out.append(path)
    for p in out:
        print(f'wrote {p}')
    return pdf


def wilson(k, n, z=1.96):
    """Wilson score interval -- the normal approximation misbehaves near 0 and 1."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def mean_ci(values, confidence=0.95):
    """Mean and half-width of the t-based CI across seeds. n<2 -> half-width 0."""
    import statistics as st
    n = len(values)
    if n < 2:
        return (float(values[0]) if n else 0.0), 0.0
    # t critical values for 95%, df = n-1, for the seed counts we actually run.
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262}.get(n - 1, 1.96)
    return st.mean(values), tcrit * st.stdev(values) / (n ** 0.5)
