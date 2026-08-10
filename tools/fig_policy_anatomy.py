"""Figure F4 -- what the LatentFlowPSM policy actually does when deployed.

(a) Antmaze rollout gallery: xy trajectories coloured by time, successes solid and
    failures faded, so the reader sees the behaviour rather than a success rate.
(b) The support claim, measured: the distribution of ||u||^2 for the latents the actor
    emits during those rollouts, against the dataset latents it was trained on and the
    chi^2_{d_a} prior. "Every executed action is a flow decode of a typical latent" is
    checkable, so it is checked.
(c) Calibration of the deployed policy: predicted psi(s, w, u)^T w at the first state of
    each episode against what that episode actually returned. Reported as AUC against
    binary success, because failed episodes all return the same value and a rank
    correlation over that tie mass is not interpretable.

Run (after tools/viz_policy_rollouts.py): .venv/bin/python tools/fig_policy_anatomy.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.figstyle import INK, INK_MUTED, LOG_DIR, PALETTE, save, use_style

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

CUBE_SEEDS = [0, 1, 2]
NPZ_DATASET = {
    'cube': '/data-local/amsks/PSMFLows/preimages_cube_single_a20_n200.npz',
    'antmaze': '/data-local/amsks/PSMFLows/preimages_antmaze_medium_a20_n200.npz',
}


def chi2_pdf(x, k):
    from math import lgamma, log
    with np.errstate(divide='ignore'):
        return np.exp((k / 2 - 1) * np.log(np.maximum(x, 1e-12)) - x / 2
                      - (k / 2) * log(2.0) - lgamma(k / 2))


def _auc(scores, pos):
    """P(score of a positive > score of a negative), ties counted as half."""
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if not n_pos or not n_neg:
        return None
    order = scores.argsort(kind='stable')
    r = np.empty(len(scores), dtype=np.float64)
    r[order] = np.arange(len(scores), dtype=np.float64)
    srt = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    r = r + 1.0
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _auc_ci(scores, pos, n_boot=4000, seed=0):
    """Percentile bootstrap over episodes. Seeded, so the figure is reproducible."""
    point = _auc(scores, pos)
    if point is None:
        return None, None, None
    rng = np.random.default_rng(seed)
    n = len(scores)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = _auc(scores[idx], pos[idx])
        if a is not None:
            vals.append(a)
    if not vals:
        return point, None, None
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load(name):
    p = f'{LOG_DIR}/{name}.json'
    if not os.path.exists(p):
        return None, None
    with open(p) as f:
        rep = json.load(f)
    npz = p.replace('.json', '.npz')
    return rep, (np.load(npz) if os.path.exists(npz) else None)


def main():
    use_style()
    report = {'inputs': {}, 'panels': {}, 'missing': []}

    am_rep, am_npz = load('rollouts_antmaze_sd0')
    cube = [(s,) + load(f'rollouts_cube_sd{s}') for s in CUBE_SEEDS]
    cube = [(s, r, z) for s, r, z in cube if r is not None]
    if am_rep is None:
        report['missing'].append('rollouts_antmaze_sd0')
    report['missing'] += [f'rollouts_cube_sd{s}' for s in CUBE_SEEDS
                          if s not in [c[0] for c in cube]]

    fig = plt.figure(figsize=(6.6, 2.5))
    gs = fig.add_gridspec(1, 3, wspace=0.38, width_ratios=[1.0, 1.05, 1.0])

    # ------------------------------------------------------------------- (a) gallery
    ax = fig.add_subplot(gs[0, 0])
    if am_npz is not None:
        xy = am_npz['xy']
        lens = am_npz['episode_lengths']
        succ = am_npz['episode_success']
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        for i, (a, n) in enumerate(zip(starts, lens)):
            traj = xy[a:a + n]
            pts = traj.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            good = succ[i] > 0.5
            lc = LineCollection(segs, cmap='viridis',
                                array=np.linspace(0, 1, len(segs)),
                                linewidths=1.0 if good else 0.5,
                                alpha=0.95 if good else 0.30, zorder=3 if good else 2)
            ax.add_collection(lc)
        ax.scatter(xy[starts, 0], xy[starts, 1], s=8, color=INK, zorder=5,
                   label='start', linewidths=0)
        ax.autoscale_view()
        ax.set_aspect('equal', adjustable='datalim')
        n_ok = int((succ > 0.5).sum())
        ax.set_title(f'(a) antmaze rollouts\n{n_ok}/{len(succ)} reach the goal',
                     loc='left', fontsize=7)
        ax.text(0.02, 0.02, 'bold = success, faded = failure\ncolour = time',
                transform=ax.transAxes, fontsize=5, color=INK_MUTED)
        report['panels']['a'] = {
            'n_episodes': int(len(succ)), 'n_success': n_ok,
            'success_rate': float((succ > 0.5).mean()),
            'mean_episode_length': float(lens.mean()),
        }
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    # ------------------------------------------------------------------- (b) support
    ax = fig.add_subplot(gs[0, 1])
    pb = {}
    series = []
    if cube:
        series.append(('cube', np.concatenate([z['latents'] for _, _, z in cube
                                               if z is not None])))
    if am_npz is not None:
        series.append(('antmaze', am_npz['latents']))
    for i, (env, lat) in enumerate(series):
        d_a = lat.shape[1]
        n2 = (lat.astype(np.float64) ** 2).sum(1)
        hi = max(np.percentile(n2, 99.5), 2.5 * d_a)
        ax.hist(n2, bins=70, range=(0, hi), density=True, histtype='step',
                color=PALETTE[i], linewidth=1.3, label=f'{env}: actor latents')
        with np.load(NPZ_DATASET[env]) as z:
            dn2 = (z['noise_preimage_point'].astype(np.float64) ** 2).sum(1)
        ax.hist(dn2, bins=70, range=(0, hi), density=True, histtype='step',
                color=PALETTE[i], linewidth=0.8, linestyle='--', alpha=0.8,
                label=f'{env}: dataset latents')
        grid = np.linspace(1e-3, hi, 300)
        ax.plot(grid, chi2_pdf(grid, d_a), color=PALETTE[i], linewidth=0.7,
                linestyle=':', alpha=0.9)
        pb[env] = {
            'd_a': d_a, 'n_steps': int(lat.shape[0]),
            'actor_mean_norm2': float(n2.mean()),
            'dataset_mean_norm2': float(dn2.mean()),
            'prior_expected_norm2': d_a,
            'actor_median_norm2': float(np.median(n2)),
        }
    ax.set_xlabel('$\\|u\\|^2$')
    ax.set_ylabel('density')
    ax.set_title('(b) the actor stays inside\nthe latent prior', loc='left', fontsize=7)
    ax.legend(fontsize=4.5, handletextpad=0.4, labelspacing=0.2, loc='upper right')
    report['panels']['b'] = pb

    # --------------------------------------------------------------- (c) calibration
    ax = fig.add_subplot(gs[0, 2])
    pc = {}
    for i, (label, reps) in enumerate([
            ('cube', [r for _, r, _ in cube]),
            ('antmaze', [am_rep] if am_rep else [])]):
        if not reps:
            continue
        pred = np.concatenate([np.array(r['calibration']['predicted_at_s0']) for r in reps])
        succ = np.concatenate([np.array([e['success'] > 0.5 for e in r['episodes']])
                               for r in reps])
        # Standardise per env: the score's scale is arbitrary, its ordering is not.
        z = (pred - pred.mean()) / (pred.std() + 1e-12)
        rng = np.random.default_rng(0)
        jitter = (rng.random(len(z)) - 0.5) * 0.28
        ax.scatter(z, succ.astype(float) + jitter, s=6, color=PALETTE[i], alpha=0.55,
                   linewidths=0, label=label)
        aucs = [r['calibration'].get('auc_predicted_vs_success') for r in reps]
        aucs = [a for a in aucs if a is not None]
        # Pooled AUC with a bootstrap interval. The question this panel asks is whether
        # the value beats chance at all, and a point estimate off ~40 successes cannot
        # answer it -- so the interval is computed, not omitted.
        pooled_auc, auc_lo, auc_hi = _auc_ci(pred, succ)
        pc[label] = {
            'auc_per_seed': aucs,
            'auc_pooled': pooled_auc, 'auc_ci95': [auc_lo, auc_hi],
            'beats_chance': (auc_lo is not None and auc_lo > 0.5),
            'n_episodes': int(len(succ)), 'n_success': int(succ.sum()),
            'spearman_per_seed': [r['calibration'].get('spearman_predicted_vs_realised')
                                  for r in reps],
        }
    ax.axhline(0.5, color=INK_MUTED, lw=0.5, linestyle=':')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['fail', 'success'])
    ax.set_xlabel('predicted $\\psi^\\top w$ at $s_0$ (std.)')
    ax.set_title('(c) does the value predict\nthe outcome?', loc='left', fontsize=7)
    txt = '\n'.join(
        f"{k}: AUC {v['auc_pooled']:.2f} [{v['auc_ci95'][0]:.2f}, {v['auc_ci95'][1]:.2f}]"
        for k, v in pc.items() if v['auc_pooled'] is not None)
    if txt:
        ax.text(0.03, 0.55, txt + '\n(0.5 = chance)', transform=ax.transAxes,
                fontsize=5.5, color=INK)
    ax.legend(fontsize=5, loc='center right', handletextpad=0.3)
    report['panels']['c'] = pc

    if report['missing']:
        print('MISSING: ' + ', '.join(report['missing']))
    save(fig, 'fig_policy_anatomy', report)


if __name__ == '__main__':
    main()
