"""Figure F2 -- the flow fits the behaviour data and inverts cleanly.

Three panels:
  (a) Fidelity against a random-flow control: RBF-MMD and mode-histogram TV between
      dataset actions and flow samples at the same states. The control is the honest
      scale -- an MMD is only small relative to what an untrained flow scores.
  (b) Inversion typicality: the distribution of ||u*||^2 against the chi^2_{d_a} density
      it should follow if the inverse lands in the prior's typical set, plus the
      effective sample size of the mixture posterior against its threshold of 20.
  (c) Round-trip error: ||G(s, u*) - a||, the residual of decoding the recovered latent.

Read from the D1 JSON reports and the preimage npz. The point of the figure is that
whatever fails downstream, it is not inversion quality.

Run: .venv/bin/python tools/fig_flow_fit.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.figstyle import INK, INK_MUTED, LOG_DIR, PALETTE, save, use_style

import matplotlib.pyplot as plt

ENVS = ['pointmaze', 'cube', 'antmaze']
NPZ = {
    'pointmaze': '/data-local/amsks/PSMFLows/preimages_pointmaze_medium_a20_n200.npz',
    'cube': '/data-local/amsks/PSMFLows/preimages_cube_single_a20_n200.npz',
    'antmaze': '/data-local/amsks/PSMFLows/preimages_antmaze_medium_a20_n200.npz',
}
ESS_THRESHOLD = 20.0


def chi2_pdf(x, k):
    """chi^2_k density without pulling in scipy."""
    from math import lgamma, log
    with np.errstate(divide='ignore'):
        logp = ((k / 2 - 1) * np.log(np.maximum(x, 1e-12)) - x / 2
                - (k / 2) * log(2.0) - lgamma(k / 2))
    return np.exp(logp)


def main():
    use_style()
    report = {'inputs': {}, 'panels': {}}

    d1 = {}
    for env in ENVS:
        for tag in ('', '_random'):
            p = f'{LOG_DIR}/d1_{env}{tag}.json'
            if os.path.exists(p):
                with open(p) as f:
                    d1[f'{env}{tag}'] = json.load(f)
                report['inputs'][f'd1_{env}{tag}'] = p
    missing = [f'{e}{t}' for e in ENVS for t in ('', '_random') if f'{e}{t}' not in d1]
    if missing:
        print(f'WARNING: missing D1 reports, panel (a) will be partial: {missing}')

    stats = {}
    for env in ENVS:
        with np.load(NPZ[env]) as z:
            u = z['noise_preimage_point'].astype(np.float64)
            stats[env] = {
                'norm2': (u ** 2).sum(1),
                'ess': z['preimage_ess'].astype(np.float64),
                'roundtrip': z['preimage_roundtrip'].astype(np.float64),
                'd_a': u.shape[1],
                'n': u.shape[0],
            }
        report['inputs'][f'npz_{env}'] = NPZ[env]

    fig = plt.figure(figsize=(6.4, 2.5))
    gs = fig.add_gridspec(1, 3, wspace=0.42)

    # ------------------------------------------------------------------ (a) fidelity
    ax = fig.add_subplot(gs[0, 0])
    width = 0.36
    xs = np.arange(len(ENVS))
    pa = {}
    for i, (metric, offset, hatch) in enumerate(
            [('mmd_rbf', -width / 2, None), ('mode_hist_tv', width / 2, '///')]):
        trained = [d1.get(e, {}).get(metric, np.nan) for e in ENVS]
        control = [d1.get(f'{e}_random', {}).get(metric, np.nan) for e in ENVS]
        ax.bar(xs + offset, trained, width * 0.92, color=PALETTE[i], hatch=hatch,
               edgecolor='white', linewidth=0.8,
               label=f"{'MMD' if metric == 'mmd_rbf' else 'mode TV'} (trained)")
        # The control is a marker, not a bar: it is a reference level, not a peer.
        ax.scatter(xs + offset, control, marker='_', s=90, linewidths=1.4,
                   color=INK, zorder=4,
                   label='random-flow control' if i == 0 else None)
        pa[metric] = {'trained': trained, 'random_control': control}
    ax.set_yscale('log')
    ax.set_xticks(xs)
    ax.set_xticklabels(ENVS, fontsize=6)
    ax.set_ylabel('distance to data (log)')
    ax.set_title('(a) flow vs a random-flow\ncontrol', loc='left', fontsize=7)
    ax.legend(fontsize=5, loc='lower right', handletextpad=0.3, labelspacing=0.25)
    report['panels']['a'] = pa

    # ---------------------------------------------------------------- (b) typicality
    ax = fig.add_subplot(gs[0, 1])
    pb = {}
    for i, env in enumerate(ENVS):
        s = stats[env]
        hi = np.percentile(s['norm2'], 99.5)
        ax.hist(s['norm2'], bins=90, range=(0, hi), density=True, histtype='step',
                color=PALETTE[i], linewidth=1.2, label=f"{env} ($d_a$={s['d_a']})")
        grid = np.linspace(1e-3, hi, 300)
        ax.plot(grid, chi2_pdf(grid, s['d_a']), color=PALETTE[i], linewidth=0.8,
                linestyle=':', alpha=0.9)
        pb[env] = {
            'd_a': s['d_a'], 'n': int(s['n']),
            'mean_norm2': float(s['norm2'].mean()), 'expected_norm2': s['d_a'],
            'median_norm2': float(np.median(s['norm2'])),
            'mean_ess': float(s['ess'].mean()),
            'frac_ess_above_threshold': float((s['ess'] > ESS_THRESHOLD).mean()),
        }
    ax.set_xlabel('$\\|u^\\star\\|^2$')
    ax.set_ylabel('density')
    ax.set_title('(b) inverted latents are typical\n(dotted: $\\chi^2_{d_a}$)',
                 loc='left', fontsize=7)
    ax.legend(fontsize=5, handletextpad=0.4, labelspacing=0.25)

    # The mixture arm's ESS is the one place this pipeline is not clean; say so on the
    # figure rather than only in the text.
    worst = min(ENVS, key=lambda e: pb[e]['frac_ess_above_threshold'])
    ax.text(0.97, 0.55,
            f"mixture arm, {worst}:\nESS {pb[worst]['mean_ess']:.1f}, only "
            f"{100 * pb[worst]['frac_ess_above_threshold']:.0f}% > {ESS_THRESHOLD:.0f}"
            "\npoint arm used throughout",
            transform=ax.transAxes, ha='right', va='top', fontsize=5, color=INK_MUTED)
    report['panels']['b'] = pb
    report['panels']['b']['ess_threshold'] = ESS_THRESHOLD

    # ------------------------------------------------------------------ (c) roundtrip
    ax = fig.add_subplot(gs[0, 2])
    pc = {}
    for i, env in enumerate(ENVS):
        r = np.sort(stats[env]['roundtrip'])
        cdf = np.arange(1, len(r) + 1) / len(r)
        step = max(1, len(r) // 4000)   # thin for a vector figure that stays small
        ax.plot(r[::step], cdf[::step], color=PALETTE[i], label=env)
        pc[env] = {
            'median': float(np.median(r)), 'p95': float(np.percentile(r, 95)),
            'p99': float(np.percentile(r, 99)), 'max': float(r.max()),
        }
    ax.set_xscale('log')
    ax.set_xlabel('$\\|G(s, u^\\star) - a\\|$')
    ax.set_ylabel('fraction of transitions')
    ax.set_title('(c) round-trip error', loc='left', fontsize=7)
    ax.legend(fontsize=5, loc='upper left', handletextpad=0.4, labelspacing=0.25)
    report['panels']['c'] = pc

    save(fig, 'fig_flow_fit', report)


if __name__ == '__main__':
    main()
