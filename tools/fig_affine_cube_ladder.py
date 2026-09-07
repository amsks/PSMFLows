"""Three-seed 500-episode checkpoint ladder for the affine-strict arm on cube.

Successor to the ad-hoc 2026-09-05 `affine-cube-collapse` figure: same JSON schema (runs /
note / series / eval500) plus an `aggregate` block, but every 50k checkpoint from 50k to
500k is now covered for three seeds instead of a partial grid for two.

The point of the figure is the SHAPE of the per-checkpoint series, not its endpoint: this
arm swings by +/-0.3 between adjacent 50k checkpoints of one run, so a single checkpoint
of it may not be quoted on its own (docs/HANDOFF.md, 2026-09-05 and 2026-09-06 entries).
Each 500-episode point carries its own Wilson interval so the reader can see that adjacent
points are disjoint by a wide margin -- i.e. that the swing is training-time
non-stationarity, not evaluation noise.

Outputs (both under docs/figures/, NOT PAPER/ICLR/figures -- this is a lab-notebook figure):
    docs/figures/<name>.png
    docs/figures/<name>.json

Usage:
    .venv/bin/python tools/fig_affine_cube_ladder.py --logs $PSM_DATA/logs \
        --exp $PSM_DATA/exp/PSMFLows --name 2026-09-06-affine-cube-ladder
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

# The 09-04 batch has no epoch token in its basename and is restore_epoch=100000; every
# later batch carries <N>k. Both forms are listed so the ladder is complete.
EPOCH_PATTERNS = {ep: [f'eval500_affine{ep // 1000}k_strict_cube_sd{{s}}.json']
                  for ep in range(50000, 550000, 50000)}
# The 09-04 batch (seeds 0 and 1 at 100k) predates the epoch token and is restore_epoch
# 100000; seed 2 was evaluated after it became mandatory. Both names are accepted.
EPOCH_PATTERNS[100000].append('eval500_affine_strict_cube_sd{s}.json')
EPOCHS = sorted(EPOCH_PATTERNS)
BC_CONTROL = 0.072  # cube, 500 ep, frozen flow acting alone (agent=fql bc_only=true)

# Series pulled out of the run's own csv logs, exactly the columns the 09-05 JSON carried.
TRAIN_COLS = ['w_enc_spread', 'psi_q_index_spread_rel', 'psi_q_spread_rel',
              'psi_q_range_rel', 'psi_q_spread', 'psm_loss', 'psm_diag', 'psm_offdiag',
              'orth_loss', 'orth_diag', 'orth_offdiag']
EVAL_COLS = {'eval_success': 'success', 'eval_control': 'control',
             'eval_episode_length': 'episode.length'}


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
    runs, series, eval500 = {}, {}, {}
    for s in seeds:
        hits = sorted(glob.glob(os.path.join(exp, 'affine_strict_cube', f'sd{int(s):03d}_*')))
        if not hits:
            raise SystemExit(f'no run dir for seed {s} under {exp}/affine_strict_cube')
        run = hits[-1]
        runs[str(s)] = run
        tr = _read_csv(os.path.join(run, 'train.csv'), 'training/',
                       {c: c for c in TRAIN_COLS})
        ev = _read_csv(os.path.join(run, 'eval.csv'), 'evaluation/', EVAL_COLS)
        ser = {'train_steps': tr.pop('step', []), 'eval_steps': ev.pop('step', [])}
        ser.update(ev)
        ser.update(tr)
        series[str(s)] = ser

        cells = {}
        for ep in EPOCHS:
            hit = [os.path.join(logs, pat.format(s=s)) for pat in EPOCH_PATTERNS[ep]]
            hit = [h for h in hit if os.path.exists(h)]
            if not hit:
                continue
            p = hit[0]
            with open(p) as fh:
                d = json.load(fh)
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
                              'train_actor': d.get('train_actor')}
        eval500[str(s)] = cells
    return runs, series, eval500


def aggregate(eval500, seeds):
    """Per-epoch mean/std across seeds, and the two late-checkpoint pooled means."""
    per_epoch = {}
    for ep in EPOCHS:
        vals = [eval500[str(s)][str(ep)]['success'] for s in seeds
                if str(ep) in eval500[str(s)]]
        if not vals:
            continue
        per_epoch[str(ep)] = {'n_seeds': len(vals), 'mean': round(st.mean(vals), 4),
                              'std': round(st.stdev(vals), 4) if len(vals) > 1 else 0.0,
                              'values': vals}

    def pooled(lo):
        vals = [c['success'] for s in seeds for e, c in eval500[str(s)].items()
                if int(e) >= lo]
        if not vals:
            return None
        m, hw = figstyle.mean_ci(vals)
        return {'from_epoch': lo, 'n_measurements': len(vals), 'mean': round(m, 4),
                'std': round(st.stdev(vals), 4) if len(vals) > 1 else 0.0,
                'ci95_half_width': round(hw, 4),
                'min': round(min(vals), 4), 'max': round(max(vals), 4)}

    return {'per_epoch': per_epoch,
            'late_300k_500k': pooled(300000),
            'late_250k_500k': pooled(250000),
            'bc_control': BC_CONTROL}


def write_table(eval500, agg, seeds, out_md):
    """docs/tables/affine_cube_ladder.md -- generated, never hand-edited."""
    L = []
    L.append('# Affine strict on cube-single-play: the 500-episode checkpoint ladder')
    L.append('')
    L.append('Arm: `psi_form=affine policy_index=latent train_actor=false acting=gpi`')
    L.append('(the repo default `agent=psmflow` since 2026-09-04).')
    L.append('')
    L.append('GENERATED by `tools/fig_affine_cube_ladder.py`; do not hand-edit.')
    L.append('')
    L.append('Every cell is one 500-episode `tools/eval_checkpoint.py` run, success with its')
    L.append('Wilson 95% interval. Adjacent checkpoints of the SAME run have disjoint')
    L.append('intervals, so the swing is training-time non-stationarity, not eval noise.')
    L.append('')
    hdr = '| epoch | ' + ' | '.join(f'seed {s}' for s in seeds) + ' | mean +/- std (across seeds) |'
    L.append(hdr)
    L.append('|---' * (len(seeds) + 2) + '|')
    for ep in EPOCHS:
        cells = []
        for s in seeds:
            c = eval500[str(s)].get(str(ep))
            cells.append('—' if c is None
                         else f"{c['success']:.3f} [{c['wilson95'][0]:.3f}, {c['wilson95'][1]:.3f}]")
        a = agg['per_epoch'].get(str(ep))
        agg_cell = '—' if a is None else f"{a['mean']:.3f} +/- {a['std']:.3f} (n={a['n_seeds']})"
        L.append(f'| {ep // 1000}k | ' + ' | '.join(cells) + f' | {agg_cell} |')
    L.append('')
    L.append('## Late-checkpoint pooled means (checkpoint x seed measurements)')
    L.append('')
    L.append('| window | n | mean | std | 95% CI half-width | min | max |')
    L.append('|---|---|---|---|---|---|---|')
    for key, label in (('late_300k_500k', '300k-500k'), ('late_250k_500k', '250k-500k')):
        b = agg.get(key)
        if b:
            L.append(f"| {label} | {b['n_measurements']} | {b['mean']:.3f} | {b['std']:.3f} | "
                     f"+/-{b['ci95_half_width']:.3f} | {b['min']:.3f} | {b['max']:.3f} |")
    L.append(f"| BC control (frozen flow alone, 500 ep) | — | {agg['bc_control']:.3f} | — | — | — | — |")
    L.append('')
    L.append('The pooled rows treat each (checkpoint, seed) 500-episode measurement as one')
    L.append('draw; the interval is a t-interval over those draws, i.e. it describes the')
    L.append('spread over checkpoints as well as over seeds. That is the only honest way to')
    L.append('quote this arm -- no single checkpoint of it is reproducible to +/-0.1.')
    L.append('')
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, 'w') as f:
        f.write('\n'.join(L))
    print(f'wrote {out_md}')


def plot(runs, series, eval500, agg, seeds, out_png):
    figstyle.use_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    ax = axes[0]
    for i, s in enumerate(seeds):
        cells = eval500[str(s)]
        xs = [ep for ep in EPOCHS if str(ep) in cells]
        ys = [cells[str(ep)]['success'] for ep in xs]
        lo = [ys[j] - cells[str(ep)]['wilson95'][0] for j, ep in enumerate(xs)]
        hi = [cells[str(ep)]['wilson95'][1] - ys[j] for j, ep in enumerate(xs)]
        ax.errorbar([x / 1000 for x in xs], ys, yerr=[lo, hi], marker='o', capsize=2,
                    color=figstyle.PALETTE[i], label=f'seed {s}')
    ax.axhline(BC_CONTROL, color=figstyle.INK_MUTED, ls='--', lw=1.0)
    ax.text(495, BC_CONTROL + 0.018, 'BC control 0.072', color=figstyle.INK_MUTED,
            fontsize=6, ha='right')
    ax.set_xlabel('checkpoint (k steps)')
    ax.set_ylabel('success (500 episodes)')
    ax.set_title('affine strict, cube: 500-ep ladder')
    ax.set_ylim(0, 0.85)
    ax.legend()

    ax = axes[1]
    for i, s in enumerate(seeds):
        ser = series[str(s)]
        if ser.get('eval_steps') and ser.get('eval_success'):
            ax.plot([x / 1000 for x in ser['eval_steps']], ser['eval_success'],
                    marker='.', color=figstyle.PALETTE[i], alpha=0.8, label=f'seed {s}')
    ax.axhline(BC_CONTROL, color=figstyle.INK_MUTED, ls='--', lw=1.0)
    ax.set_xlabel('step (k)')
    ax.set_ylabel('success (50 episodes, in-loop)')
    ax.set_title('in-loop eval (noisy: 95% CI ~ +/-0.115)')
    ax.set_ylim(0, 0.85)
    ax.legend()

    ax = axes[2]
    for i, s in enumerate(seeds):
        ser = series[str(s)]
        if ser.get('train_steps') and ser.get('w_enc_spread'):
            n = min(len(ser['train_steps']), len(ser['w_enc_spread']))
            ax.plot([x / 1000 for x in ser['train_steps'][:n]], ser['w_enc_spread'][:n],
                    color=figstyle.PALETTE[i], label=f'seed {s}')
    ax.set_xlabel('step (k)')
    ax.set_ylabel('mean pairwise ||w(u_i) - w(u_j)||')
    ax.set_title('policy-encoder spread (collapse -> 0)')
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f'wrote {out_png}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', required=True, help='dir of eval500_*.json reports')
    ap.add_argument('--exp', required=True, help='experiment root holding affine_strict_cube/')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--name', default='2026-09-06-affine-cube-ladder')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--table', default='affine_cube_ladder.md')
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(',')]
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    runs, series, eval500 = collect(args.logs, args.exp, seeds)
    agg = aggregate(eval500, seeds)
    report = {'runs': runs,
              'note': ('affine_strict cube, 3 seeds; the ladder is 500-episode '
                       'tools/eval_checkpoint.py evals at every 50k checkpoint. In-loop '
                       'eval is 50 episodes (+/-0.115); train.csv logged every 5k.'),
              'series': series, 'eval500': eval500, 'aggregate': agg}
    with open(os.path.join(out_dir, f'{args.name}.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print(f'wrote {os.path.join(out_dir, args.name)}.json')
    plot(runs, series, eval500, agg, seeds, os.path.join(out_dir, f'{args.name}.png'))
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    write_table(eval500, agg, seeds, os.path.join(repo, 'docs', 'tables', args.table))


if __name__ == '__main__':
    main()
