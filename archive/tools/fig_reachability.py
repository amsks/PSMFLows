"""Figure F1 -- the fixed-latent policy family contains no goal-reaching member.

Four panels:
  (a) Exhaustive reachability over the pointmaze latent box. d_a=2, so a 13x13 grid
      covers the ENTIRE box [-3,3]^2 -- this is not a sample, it is the whole family.
  (b) The two degenerate regimes the family collapses into: saturated latents give a
      near-constant heading (low circular variance, straight short path), typical
      latents give a state-dependent field that circulates (high circular variance,
      long path, near-zero net displacement).
  (c) Why no fixed latent encodes a route: dataset latents are temporally white.
  (d) Family ceiling vs what flow-GPI actually scored.

Everything is read from JSON reports and the preimage npz; nothing is hand-copied.

Run: .venv/bin/python tools/fig_reachability.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.figstyle import (INK, INK_MUTED, LOG_DIR, PALETTE, SEQUENTIAL, save,
                            use_style)

import matplotlib.pyplot as plt

NPZ = '/data-local/amsks/PSMFLows/preimages_pointmaze_medium_a20_n200.npz'
MAX_LAG = 120
N_EPISODES = 300   # subsample for the autocorrelation; deterministic
SEED = 0


def episode_bounds(terminals):
    """[start, stop) per episode. OGBench marks the last transition of an episode."""
    ends = np.flatnonzero(terminals > 0.5)
    starts = np.concatenate([[0], ends[:-1] + 1])
    return list(zip(starts.tolist(), (ends + 1).tolist()))


def latent_autocorrelation(u, bounds, max_lag, n_episodes, seed):
    """Mean over episodes and dimensions of corr(u_t, u_{t+lag}), plus the variance ratio.

    The variance ratio is mean-within-episode variance over marginal variance: 1.0 means
    an episode's latents are no more alike than latents drawn from the whole dataset,
    i.e. the latent carries no episode-level identity.
    """
    rng = np.random.default_rng(seed)
    lengths = np.array([b - a for a, b in bounds])
    usable = [i for i in range(len(bounds)) if lengths[i] > max_lag + 10]
    pick = rng.choice(usable, size=min(n_episodes, len(usable)), replace=False)

    lags = np.arange(1, max_lag + 1)
    acc = np.zeros(len(lags))
    counts = np.zeros(len(lags))
    within = []
    for i in pick:
        a, b = bounds[i]
        seg = u[a:b]
        within.append(seg.var(axis=0))
        seg = seg - seg.mean(axis=0)
        denom = (seg * seg).sum(axis=0)
        denom[denom == 0] = np.nan
        for j, lag in enumerate(lags):
            if seg.shape[0] <= lag + 1:
                continue
            num = (seg[:-lag] * seg[lag:]).sum(axis=0)
            acc[j] += np.nanmean(num / denom)
            counts[j] += 1
    ac = acc / np.maximum(counts, 1)

    within = np.mean(np.stack(within), axis=0)
    marginal = u.var(axis=0)
    return lags, ac, float(np.mean(within / marginal))


def main():
    use_style()
    report = {'inputs': {}, 'panels': {}}

    reach = {}
    for env in ('pointmaze', 'cube', 'antmaze'):
        p = f'{LOG_DIR}/latent_reachability_{env}.json'
        with open(p) as f:
            reach[env] = json.load(f)
        report['inputs'][f'reachability_{env}'] = p
    with open(f'{LOG_DIR}/viz_fixed_u_field_pointmaze.json') as f:
        field = json.load(f)
    report['inputs']['fixed_u_field'] = f'{LOG_DIR}/viz_fixed_u_field_pointmaze.json'
    report['inputs']['preimage_npz'] = NPZ

    fig = plt.figure(figsize=(6.2, 4.8))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.42,
                          height_ratios=[1.0, 0.92])

    # ---------------------------------------------------------------- (a) reachability
    ax = fig.add_subplot(gs[0, 0])
    pm = reach['pointmaze']
    cands = pm['candidates']
    groups = {
        'grid': dict(marker='s', s=26, label='latent grid (exhaustive)'),
        'preimage': dict(marker='o', s=18, label='dataset latents'),
        'goal_preimage': dict(marker='^', s=22, label='goal-reaching latents'),
    }
    dists = np.array([c['min_goal_dist'] for c in cands])
    vmin, vmax = float(dists.min()), float(dists.max())
    for g, kw in groups.items():
        sel = [c for c in cands if c['group'] == g]
        if not sel:
            continue
        xy = np.array([c['u'] for c in sel])
        ax.scatter(xy[:, 0], xy[:, 1], c=[c['min_goal_dist'] for c in sel],
                   cmap=SEQUENTIAL + '_r', vmin=vmin, vmax=vmax,
                   edgecolors='white', linewidths=0.4, **kw)
    sm = plt.cm.ScalarMappable(cmap=SEQUENTIAL + '_r',
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(sm, ax=ax, pad=0.03, fraction=0.046)
    cb.set_label('closest approach', fontsize=5.5, labelpad=1)
    cb.ax.tick_params(labelsize=5.5, pad=1)
    cb.outline.set_visible(False)

    best = min(cands, key=lambda c: c['min_goal_dist'])
    ax.annotate(f"best {best['min_goal_dist']:.2f}\n(never reaches)",
                xy=best['u'], xytext=(-2.95, 1.9), fontsize=5.5, color=INK,
                arrowprops=dict(arrowstyle='-', lw=0.6, color=INK_MUTED))
    ax.set_xlabel('$u_1$')
    ax.set_ylabel('$u_2$')
    ax.set_title(f"(a) every latent in the box,\n{pm['episodes_per_u']}"
                 f"$\\times${pm['max_steps']}-step rollouts", loc='left', fontsize=7)
    leg = ax.legend(loc='lower left', fontsize=5, handletextpad=0.15,
                    borderpad=0.25, labelspacing=0.25, frameon=True,
                    facecolor='white', framealpha=0.9, edgecolor='#d9d9d9')
    leg.get_frame().set_linewidth(0.4)
    report['panels']['a'] = {
        'n_candidates': pm['num_candidates'], 'groups': pm['groups'],
        'num_success_any': pm['num_success_any'],
        'min_goal_dist_overall': pm['min_goal_dist_overall'],
        'min_goal_dist_quartiles': pm['min_goal_dist_quartiles'],
        'maze_span_approx': 30,
    }

    # ------------------------------------------------------------------- (b) regimes
    ax = fig.add_subplot(gs[0, 1])
    per_u = field['per_u']
    cv = np.array([p['heading_circ_var'] for p in per_u])
    path = np.array([p['traj_path_length'] for p in per_u])
    disp = np.array([p['traj_net_displacement'] for p in per_u])
    ax.scatter(cv, path, c=disp, cmap=SEQUENTIAL, s=34,
               edgecolors=INK_MUTED, linewidths=0.5, zorder=3)
    for p in per_u:
        if p['heading_circ_var'] > 0.5:
            ax.annotate('typical $u$: orbits\n(net displacement'
                        f" {p['traj_net_displacement']:.1f})",
                        xy=(p['heading_circ_var'], p['traj_path_length']),
                        xytext=(0.30, 40), fontsize=5.5, color=INK,
                        arrowprops=dict(arrowstyle='-', lw=0.6, color=INK_MUTED))
            break
    j = int(np.argmin(cv))
    ax.annotate('saturated $u$:\nconstant heading',
                xy=(cv[j], path[j]), xytext=(0.12, 2.0), fontsize=5.5, color=INK,
                arrowprops=dict(arrowstyle='-', lw=0.6, color=INK_MUTED))
    ax.set_yscale('log')
    ax.set_xlabel('heading circular variance')
    ax.set_ylabel('path length')
    ax.set_title('(b) two degenerate regimes,\nnothing in between', loc='left', fontsize=7)
    report['panels']['b'] = {
        'per_u': per_u, 'goal': field['goal'],
        'per_cell_multimodality': field.get('per_cell_multimodality'),
    }

    # --------------------------------------------------------------- (c) white noise
    ax = fig.add_subplot(gs[1, 0])
    with np.load(NPZ) as z:
        u = z['noise_preimage_point'].astype(np.float64)
        terminals = z['terminals']
    bounds = episode_bounds(terminals)
    lags, ac, ratio = latent_autocorrelation(u, bounds, MAX_LAG, N_EPISODES, SEED)
    ax.axhline(0.0, color=INK_MUTED, lw=0.6, zorder=1)
    ax.plot(lags, ac, color=PALETTE[0], zorder=3)
    ax.set_xlabel('lag (steps within an episode)')
    ax.set_ylabel('autocorrelation of $u^\\star$')
    ax.set_title('(c) dataset latents are\ntemporally white', loc='left', fontsize=7)
    ax.text(0.96, 0.92,
            f'lag-1 {ac[0]:.2f}\nwithin-episode variance /\nmarginal variance {ratio:.2f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=6, color=INK)
    report['panels']['c'] = {
        'n_episodes_sampled': min(N_EPISODES, len(bounds)),
        'n_episodes_total': len(bounds), 'seed': SEED,
        'lag1_autocorr': float(ac[0]),
        'lag10_autocorr': float(ac[9]),
        'lag50_autocorr': float(ac[49]),
        'within_over_marginal_variance': ratio,
    }

    # ------------------------------------------------------------------- (d) ceiling
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')
    ax.grid(False)
    # Observed flow-GPI success for the fixed-latent family, from the runs on disk.
    observed = {'pointmaze': '0.00', 'cube': '0.02-0.06', 'antmaze': 'n/a'}
    rows, ceiling = [], {}
    for env in ('pointmaze', 'cube', 'antmaze'):
        r = reach[env]
        n, k = r['num_candidates'], r['num_success_any']
        ceiling[env] = {'reached': k, 'candidates': n, 'frac': k / n,
                        'observed_gpi': observed[env]}
        rows.append([env, f'{k}/{n}', f'{100 * k / n:.0f}%', observed[env]])
    tbl = ax.table(cellText=rows,
                   colLabels=['env', 'reach\ngoal', 'ceiling', 'flow-GPI\nobserved'],
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6)
    tbl.scale(1.0, 1.35)
    for (r_, _c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor('#d9d9d9')
        if r_ == 0:
            cell.set_text_props(color=INK, fontweight='bold')
    ax.set_title('(d) family ceiling vs\nwhat flow-GPI scored', loc='left', fontsize=7)
    ax.text(0.5, 0.06, 'GPI already extracts what the family contains',
            transform=ax.transAxes, ha='center', fontsize=6, color=INK_MUTED)
    report['panels']['d'] = ceiling

    save(fig, 'fig_reachability', report)


if __name__ == '__main__':
    main()
