"""Learning-curve figure: 500-episode success vs. training checkpoint, per env.

One panel per env (cube, antmaze), aggregated across seeds. Series:

1. Affine PSMFlow (ours) -- every `eval500_affine*strict_{env}_sd*.json` in the logs dir.
   The epoch comes from each JSON's own `restore_epoch`, never the filename: the 100k batch
   predates the epoch token in the basename (`eval500_affine_strict_cube_sd{0,1}.json`), and
   the JSON's own `seed` field is a known bug -- it always records the *eval* seed (0), not
   the training seed -- so seed identity is instead parsed from the `_sd<N>` filename suffix
   (docs/HANDOFF.md, 2026-09-06 entry). (epoch, seed) pairs that collide across two filename
   spellings are de-duplicated by file mtime, newest wins.
2. BC control -- horizontal line from psm-data/evals/bc_{env}.json, with its Wilson band.
3. Cube-only baselines -- FB no-BC and raw-action PSM no-BC, both ~0 everywhere.
4. A dotted "late running mean" -- the cumulative pooled mean of the affine series computed
   over epochs >= 250k, annotated at the 300k-500k pooled value.

Run:
    .venv/bin/python tools/fig_affine_learning_curve.py
    .venv/bin/python tools/fig_affine_learning_curve.py --logs $PSM_DATA/logs \
        --evals-dir $PSM_DATA/evals --name 2026-09-06-affine-learning-curve
"""
import argparse
import glob
import json
import os
import re
import statistics as st

import matplotlib

matplotlib.use('Agg')

import figstyle
import matplotlib.pyplot as plt

ENVS = ['cube', 'antmaze']
ENV_TITLE = {'cube': 'cube-single-play', 'antmaze': 'antmaze-medium-navigate'}
LATE_FROM = 250000
FINAL_WINDOW = (300000, 500000)
Z = 1.96  # figure/table CI half-width uses the normal approximation, per spec.

SD_RE = re.compile(r'_sd(\d+)\.json$')


def _num_success_episodes(d):
    """A couple of legacy JSONs (archived PSM-raw arm) use successes/episodes instead."""
    ns = d.get('num_success', d.get('successes'))
    ne = d.get('num_episodes', d.get('episodes'))
    return ns, ne


def _load_affine(logs, env):
    """{(epoch, seed): {success, n, json, mtime}} -- newest file wins on collision."""
    pattern = os.path.join(logs, f'eval500_affine*strict_{env}_sd*.json')
    best = {}
    for p in glob.glob(pattern):
        m = SD_RE.search(os.path.basename(p))
        if not m:
            continue
        seed = int(m.group(1))
        with open(p) as f:
            d = json.load(f)
        epoch = int(d['restore_epoch'])
        mtime = os.path.getmtime(p)
        key = (epoch, seed)
        if key in best and best[key]['mtime'] >= mtime:
            continue
        ns, ne = _num_success_episodes(d)
        best[key] = {'success': float(d['success']), 'num_success': ns, 'num_episodes': ne,
                     'json': os.path.basename(p), 'mtime': mtime}
    return best


def _load_baseline(logs, glob_pat):
    """{(epoch, seed): success} for the cube-only no-BC baselines."""
    out = {}
    rx = re.compile(r'_(\d+)k_sd(\d+)\.json$')
    for p in glob.glob(os.path.join(logs, glob_pat)):
        m = rx.search(os.path.basename(p))
        if not m:
            continue
        epoch, seed = int(m.group(1)) * 1000, int(m.group(2))
        with open(p) as f:
            d = json.load(f)
        if int(d.get('restore_epoch', epoch)) != epoch:
            epoch = int(d['restore_epoch'])
        out[(epoch, seed)] = float(d['success'])
    return out


def _load_bc(evals_dir, env):
    path = os.path.join(evals_dir, f'bc_{env}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return {'success': float(d['success']), 'wilson95': d.get('wilson95'),
            'json': os.path.basename(path)}


def _mean_ci(vals):
    """(mean, 95% half-width) via mean +/- 1.96*std/sqrt(n); n=1 -> half-width 0."""
    n = len(vals)
    m = st.mean(vals)
    if n < 2:
        return m, 0.0
    return m, Z * st.stdev(vals) / (n ** 0.5)


def per_epoch_stats(cells):
    """epoch -> {mean, half_width, std, n, values} across seeds, sorted by epoch."""
    by_epoch = {}
    for (epoch, seed), c in cells.items():
        by_epoch.setdefault(epoch, []).append(c['success'])
    out = {}
    for epoch, vals in by_epoch.items():
        m, hw = _mean_ci(vals)
        out[epoch] = {'mean': m, 'half_width': hw,
                      'std': st.stdev(vals) if len(vals) > 1 else 0.0,
                      'n': len(vals), 'values': sorted(vals)}
    return out


def running_late_mean(cells, from_epoch):
    """epoch (>= from_epoch, present) -> cumulative pooled mean of all measurements so far."""
    epochs = sorted({e for (e, _s) in cells if e >= from_epoch})
    out = {}
    pool = []
    for e in epochs:
        pool.extend(c['success'] for (ep, _s), c in cells.items() if ep == e)
        out[e] = st.mean(pool)
    return out


def pooled_window(cells, lo, hi):
    vals = [c['success'] for (e, _s), c in cells.items() if lo <= e <= hi]
    if not vals:
        return None
    m, hw = _mean_ci(vals)
    return {'n': len(vals), 'mean': m, 'half_width': hw,
            'std': st.stdev(vals) if len(vals) > 1 else 0.0}


def collect(logs, evals_dir):
    data = {}
    for env in ENVS:
        data[env] = {'affine': _load_affine(logs, env)}
    data['cube']['fb_nobc'] = _load_baseline(logs, 'eval500_fb_nobc_cube_*.json')
    data['cube']['psm_raw_nobc'] = _load_baseline(logs, 'eval500_psm_raw_nobc_cube_*.json')
    data['bc'] = {env: _load_bc(evals_dir, env) for env in ENVS}
    return data


def fmt_cell(stats, epoch):
    s = stats.get(epoch)
    if s is None:
        return '--'
    return f"{s['mean']:.3f} ± {s['half_width']:.3f} (n={s['n']})"


def write_table(data, agg, out_md):
    affine_epochs = sorted({e for env in ENVS for e in agg[env]['per_epoch']})
    L = []
    L.append('# Affine PSMFlow learning curve: 500-episode success vs. checkpoint')
    L.append('')
    L.append('GENERATED by `tools/fig_affine_learning_curve.py`; do not hand-edit.')
    L.append('')
    L.append('Mean +/- 95% CI is the normal approximation `mean +/- 1.96*std/sqrt(n)` across')
    L.append('seeds at that checkpoint (n=1 -> half-width 0), matching the shaded band in the')
    L.append('figure. Cube has 3 seeds at every checkpoint; antmaze is a sparse subset --')
    L.append('see n. FB-noBC and PSM-raw-noBC are cube-only baselines (mean over seeds present).')
    L.append('')
    L.append('| epoch | cube mean ± 95% CI (n) | antmaze mean ± 95% CI (n) | '
             'FB-noBC cube | PSM-raw-noBC cube |')
    L.append('|---|---|---|---|---|')
    for epoch in affine_epochs:
        cube = fmt_cell(agg['cube']['per_epoch'], epoch)
        ant = fmt_cell(agg['antmaze']['per_epoch'], epoch)
        fb = agg['cube']['fb_nobc'].get(epoch)
        psm = agg['cube']['psm_raw_nobc'].get(epoch)
        fb_s = '--' if fb is None else f"{fb['mean']:.3f} (n={fb['n']})"
        psm_s = '--' if psm is None else f"{psm['mean']:.3f} (n={psm['n']})"
        L.append(f'| {epoch // 1000}k | {cube} | {ant} | {fb_s} | {psm_s} |')
    L.append('')
    L.append(f'## Final row: pooled {FINAL_WINDOW[0] // 1000}k-{FINAL_WINDOW[1] // 1000}k '
             '(checkpoint x seed measurements) and BC controls')
    L.append('')
    L.append('| series | n | mean ± 95% CI | BC control |')
    L.append('|---|---|---|---|')
    for env in ENVS:
        w = agg[env]['final_window']
        bc = data['bc'][env]
        w_s = '--' if w is None else f"{w['n']} | {w['mean']:.3f} ± {w['half_width']:.3f}"
        bc_s = '--' if bc is None else f"{bc['success']:.3f}"
        L.append(f'| affine PSMFlow, {env} | {w_s} | {bc_s} |')
    L.append('')
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, 'w') as f:
        f.write('\n'.join(L))
    print(f'wrote {out_md}')


def plot(data, agg, out_png):
    figstyle.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), sharey=True)

    for ax, env in zip(axes, ENVS):
        cells = data[env]['affine']
        seeds = sorted({s for (_e, s) in cells})
        stats = agg[env]['per_epoch']
        epochs = sorted(stats)

        # thin per-seed lines in the background
        for i, s in enumerate(seeds):
            xs = sorted(e for (e, ss) in cells if ss == s)
            ys = [cells[(e, s)]['success'] for e in xs]
            ax.plot([x / 1000 for x in xs], ys, lw=0.7, alpha=0.45,
                    color=figstyle.PALETTE[i % len(figstyle.PALETTE)],
                    label=f'seed {s} (raw)')

        # bold mean line + shaded CI band
        mx = [e / 1000 for e in epochs]
        my = [stats[e]['mean'] for e in epochs]
        lo = [stats[e]['mean'] - stats[e]['half_width'] for e in epochs]
        hi = [stats[e]['mean'] + stats[e]['half_width'] for e in epochs]
        ax.plot(mx, my, lw=2.2, color=figstyle.INK, marker='o', markersize=4,
                label='affine PSMFlow, mean')
        ax.fill_between(mx, lo, hi, color=figstyle.INK, alpha=0.15, linewidth=0)
        for e in epochs:
            ax.annotate(f'n={stats[e]["n"]}', (e / 1000, stats[e]['mean']),
                        textcoords='offset points', xytext=(0, 6), fontsize=5.5,
                        color=figstyle.INK_MUTED, ha='center')

        # late running mean, dotted
        run = agg[env]['running_late']
        if run:
            rx = sorted(run)
            ax.plot([x / 1000 for x in rx], [run[x] for x in rx], ls=':', lw=1.6,
                    color=figstyle.PALETTE[1], label='late running mean (>=250k, cumulative)')

        # BC control
        bc = data['bc'][env]
        if bc is not None:
            ax.axhline(bc['success'], color=figstyle.INK_MUTED, ls='--', lw=1.1)
            if bc.get('wilson95'):
                ax.axhspan(bc['wilson95'][0], bc['wilson95'][1], color=figstyle.INK_MUTED,
                          alpha=0.12, linewidth=0)
            ax.text(mx[-1] if mx else 500, bc['success'] + 0.015,
                    f"BC control {bc['success']:.3f}", color=figstyle.INK_MUTED, fontsize=6.5,
                    ha='right')

        # cube-only baselines
        if env == 'cube':
            for key, label, color in (('fb_nobc', 'FB no-BC', figstyle.PALETTE[2]),
                                      ('psm_raw_nobc', 'PSM-raw no-BC', figstyle.PALETTE[3])):
                base = data[env][key]
                bx = sorted({e for (e, _s) in base})
                if not bx:
                    continue
                by = [st.mean(v for (e, _s), v in base.items() if e == e0) for e0 in bx]
                ax.plot([x / 1000 for x in bx], by, lw=1.3, color=color, marker='s',
                        markersize=3, label=label)

        final = agg[env]['final_window']
        final_label = ('final late mean (300k-500k) = '
                       f"{final['mean']:.3f} ± {final['half_width']:.3f} (n={final['n']})"
                       if final else 'final late mean (300k-500k) = --')
        ax.plot([], [], ' ', label=final_label)

        ax.set_xlabel('checkpoint (k steps)')
        ax.set_title(ENV_TITLE[env])
        ax.set_ylim(0, 0.8)
        ax.legend(fontsize=5.5, loc='upper left', ncol=1)

    axes[0].set_ylabel('success (500 episodes)')
    fig.suptitle('Affine PSMFlow learning curve: 500-episode success vs. checkpoint')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f'wrote {out_png}')


def build_report(data):
    agg = {}
    for env in ENVS:
        cells = data[env]['affine']
        agg[env] = {
            'per_epoch': per_epoch_stats(cells),
            'running_late': running_late_mean(cells, LATE_FROM),
            'final_window': pooled_window(cells, *FINAL_WINDOW),
        }
    agg['cube']['fb_nobc'] = per_epoch_stats(
        {(e, s): {'success': v} for (e, s), v in data['cube']['fb_nobc'].items()})
    agg['cube']['psm_raw_nobc'] = per_epoch_stats(
        {(e, s): {'success': v} for (e, s), v in data['cube']['psm_raw_nobc'].items()})
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', default='/mnt/home/amohan/psm-data/logs')
    ap.add_argument('--evals-dir', default='/mnt/home/amohan/psm-data/evals')
    ap.add_argument('--name', default='2026-09-06-affine-learning-curve')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--table', default='affine_learning_curve.md')
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out_dir or os.path.join(repo, 'docs', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    data = collect(args.logs, args.evals_dir)
    agg = build_report(data)

    def serializable(cells):
        return {f'{e}_{s}': c for (e, s), c in cells.items()}

    report = {
        'logs': args.logs, 'evals_dir': args.evals_dir,
        'note': ('epoch is read from each JSON\'s restore_epoch, never the filename; seed is '
                 'parsed from the filename\'s _sd<N> suffix because the JSON seed field is a '
                 'known bug (always 0, the eval seed). (epoch, seed) collisions across '
                 'filename spellings are resolved by mtime, newest wins.'),
        'affine_cube_cells': serializable(data['cube']['affine']),
        'affine_antmaze_cells': serializable(data['antmaze']['affine']),
        'fb_nobc_cube_cells': {f'{e}_{s}': v for (e, s), v in data['cube']['fb_nobc'].items()},
        'psm_raw_nobc_cube_cells': {f'{e}_{s}': v for (e, s), v in
                                    data['cube']['psm_raw_nobc'].items()},
        'bc_control': data['bc'],
        'aggregate': {
            env: {'per_epoch': {str(e): v for e, v in agg[env]['per_epoch'].items()},
                  'running_late_mean': {str(e): v for e, v in
                                       agg[env]['running_late'].items()},
                  'final_window_300k_500k': agg[env]['final_window']}
            for env in ENVS
        },
    }
    out_json = os.path.join(out_dir, f'{args.name}.json')
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=1)
    print(f'wrote {out_json}')

    out_png = os.path.join(out_dir, f'{args.name}.png')
    plot(data, agg, out_png)

    write_table(data, agg, os.path.join(repo, 'docs', 'tables', args.table))


if __name__ == '__main__':
    main()
