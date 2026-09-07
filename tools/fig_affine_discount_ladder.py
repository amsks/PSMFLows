"""Discount sweep: the affine-strict 500-episode ladder at gamma=0.98 vs gamma=0.99.

Pre-registration: docs/design/2026-09-07-discount-sweep.md. The 09-06 antmaze H2 test
(docs/design/2026-09-06-antmaze-failure-tests.md) found that raising `agent.discount` from
the repo default 0.98 to 0.99 moved antmaze off the floor (0.534 @100k vs a 0.081 late-ckpt
mean), while 0.995 moved it further and then collapsed to 0.000 in-loop on all three seeds.
This figure asks the same question on the two other envs with published Stage-A/B
artifacts, where the two arms make OPPOSITE predictions: cube is not horizon-starved (the
reward is reachable inside gamma=0.98's ~50-step horizon) and pointmaze is a settled
negative whose mechanism -- COMPENDIUM 4.11, no goal-reaching member in the fixed-`u`
policy family -- is upstream of anything the discount touches.

Companion of tools/fig_affine_cube_ladder.py: same JSON schema (runs / note / series /
eval500 / aggregate), same Wilson intervals per 500-episode cell, same "the shape of the
series is the point, not its endpoint" framing. The difference is that every quantity is
carried for TWO discounts and TWO envs, so the object plotted is the pair, never one arm.

Outputs (docs/figures + docs/tables, NOT PAPER/ICLR/figures -- lab-notebook artifacts):
    docs/figures/<name>.png
    docs/figures/<name>.json
    docs/tables/affine_discount_ladder.md

Usage:
    .venv/bin/python tools/fig_affine_discount_ladder.py --logs $PSM_DATA/logs \
        --exp $PSM_DATA/exp/PSMFLows --name 2026-09-07-affine-discount-ladder
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

EPOCHS = list(range(50000, 550000, 50000))

# env -> (run-group stem, BC control at 500 episodes, plot y-limit).
# Both BC numbers are the frozen Stage-A flow acting alone (agent=fql agent.bc_only=true)
# from the SAME checkpoint the agent decodes through, which is the only control this arm
# has to beat. cube: $PSM_DATA/evals/bc_cube.json. pointmaze:
# $PSM_DATA/logs/bc_control_pointmaze.json.
ENVS = {
    'cube': {'group': 'affine_strict_cube', 'bc': 0.072, 'ylim': 0.85,
             'label': 'cube-single-play (OGBench task 2)'},
    'pointmaze': {'group': 'affine_strict_pointmaze', 'bc': 0.002, 'ylim': 0.85,
                  'label': 'pointmaze-medium-navigate (task 1)'},
}

# discount key -> (group suffix, eval-json infix, plot colour index, legend label).
# gamma=0.98 is the repo default and its group/basenames carry no discount token at all;
# gamma=0.99 is this sweep. Keeping the mapping explicit rather than templated is the same
# discipline as make_tables.py: a glob that matched both would silently pool two arms.
ARMS = [
    ('0.98', {'suffix': '', 'infix': '', 'colour': 0, 'label': r'$\gamma$=0.98 (repo default)'}),
    ('0.99', {'suffix': '_g99', 'infix': '_g99', 'colour': 1, 'label': r'$\gamma$=0.99'}),
]

TRAIN_COLS = ['w_enc_spread', 'psi_q_index_spread_rel', 'psi_q_spread_rel',
              'psi_q_range_rel', 'psi_q_spread', 'psm_loss', 'psm_diag', 'psm_offdiag',
              'orth_loss', 'orth_diag', 'orth_offdiag']
EVAL_COLS = {'eval_success': 'success', 'eval_control': 'control',
             'eval_episode_length': 'episode.length'}


def _patterns(env, arm, epoch, seed):
    """Accepted basenames for one (env, discount, epoch, seed) cell, best first.

    The 09-04 cube batch predates the epoch token: those basenames are
    eval500_affine_strict_cube_sd<S>.json and are all restore_epoch=100000. Every reader
    re-checks restore_epoch against the epoch it asked for, so a mislabelled file raises
    instead of landing in the wrong row.
    """
    n = epoch // 1000
    pats = [f'eval500_affine{n}k_strict_{env}{arm["infix"]}_sd{seed}.json']
    if env == 'cube' and epoch == 100000 and not arm['infix']:
        pats.append(f'eval500_affine_strict_{env}_sd{seed}.json')
    return pats


def _read_csv(path, prefix, cols):
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
    """{env: {discount: {'runs':…, 'series':…, 'eval500':…}}}; missing cells are omitted."""
    data = {}
    for env, meta in ENVS.items():
        data[env] = {}
        for gamma, arm in ARMS:
            group = meta['group'] + arm['suffix']
            runs, series, eval500 = {}, {}, {}
            for s in seeds:
                hits = sorted(glob.glob(os.path.join(exp, group, f'sd{int(s):03d}_*')))
                if hits:
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
                    hit = [os.path.join(logs, p) for p in _patterns(env, arm, ep, s)]
                    hit = [h for h in hit if os.path.exists(h)]
                    if not hit:
                        continue
                    with open(hit[0]) as fh:
                        d = json.load(fh)
                    if int(d['restore_epoch']) != ep:
                        raise SystemExit(
                            f'{hit[0]}: restore_epoch {d["restore_epoch"]} != {ep}')
                    k, n = int(d['num_success']), int(d['num_episodes'])
                    _, lo, hi = figstyle.wilson(k, n)
                    cells[str(ep)] = {'success': float(d['success']),
                                      'wilson95': [round(lo, 4), round(hi, 4)],
                                      'num_success': k, 'num_episodes': n,
                                      'json': os.path.basename(hit[0])}
                eval500[str(s)] = cells
            data[env][gamma] = {'group': group, 'runs': runs, 'series': series,
                                'eval500': eval500}
    return data


def aggregate(data, seeds):
    for env, meta in ENVS.items():
        for gamma, _ in ARMS:
            blk = data[env][gamma]
            per_epoch = {}
            for ep in EPOCHS:
                vals = [blk['eval500'][str(s)][str(ep)]['success'] for s in seeds
                        if str(ep) in blk['eval500'].get(str(s), {})]
                if not vals:
                    continue
                m, hw = figstyle.mean_ci(vals)
                per_epoch[str(ep)] = {
                    'n_seeds': len(vals), 'mean': round(m, 4),
                    'std': round(st.stdev(vals), 4) if len(vals) > 1 else 0.0,
                    'ci95_half_width': round(hw, 4), 'values': vals}
            vals = [c['success'] for s in seeds
                    for e, c in blk['eval500'].get(str(s), {}).items() if int(e) >= 300000]
            late = None
            if vals:
                m, hw = figstyle.mean_ci(vals)
                late = {'from_epoch': 300000, 'n_measurements': len(vals),
                        'mean': round(m, 4),
                        'std': round(st.stdev(vals), 4) if len(vals) > 1 else 0.0,
                        'ci95_half_width': round(hw, 4),
                        'min': round(min(vals), 4), 'max': round(max(vals), 4)}
            # Per-seed oscillation, the second pre-registered quantity: the largest jump
            # between ADJACENT 50k checkpoints, and the range over the ten checkpoints.
            osc = {}
            for s in seeds:
                cells = blk['eval500'].get(str(s), {})
                xs = [ep for ep in EPOCHS if str(ep) in cells]
                if len(xs) < 2:
                    continue
                ys = [cells[str(ep)]['success'] for ep in xs]
                jumps = [abs(ys[i + 1] - ys[i]) for i in range(len(ys) - 1)]
                osc[str(s)] = {'n_checkpoints': len(xs), 'min': round(min(ys), 4),
                               'max': round(max(ys), 4),
                               'range': round(max(ys) - min(ys), 4),
                               'max_adjacent_jump': round(max(jumps), 4)}
            blk['aggregate'] = {'per_epoch': per_epoch, 'late_300k_500k': late,
                                'oscillation': osc, 'bc_control': meta['bc']}
    return data


def write_table(data, seeds, out_md):
    L = ['# Affine strict: the 500-episode checkpoint ladder at gamma=0.98 vs gamma=0.99',
         '',
         'Arm: `psi_form=affine policy_index=latent index_agg=max train_actor=false',
         "acting=gpi` -- the repo default `agent=psmflow`. The ONLY difference between the",
         'two columns of each block is `agent.discount`; flow checkpoint, preimages, seeds,',
         'budget and eval protocol are identical.',
         '',
         'GENERATED by `tools/fig_affine_discount_ladder.py`; do not hand-edit.',
         'Pre-registration: `docs/design/2026-09-07-discount-sweep.md`.',
         '',
         'Every cell is one 500-episode `tools/eval_checkpoint.py` run with `EVAL_WORKERS=1`',
         '(the serial path every earlier eval500 number used), success with its Wilson 95%',
         'interval.',
         '']
    for env, meta in ENVS.items():
        L.append(f'## {env} — {meta["label"]}')
        L.append('')
        L.append(f'BC control (frozen flow acting alone, 500 ep): **{meta["bc"]:.3f}**')
        L.append('')
        for gamma, arm in ARMS:
            blk = data[env][gamma]
            L.append(f'### gamma = {gamma}  (`{blk["group"]}`)')
            L.append('')
            L.append('| epoch | ' + ' | '.join(f'seed {s}' for s in seeds)
                     + ' | mean +/- std (across seeds) |')
            L.append('|---' * (len(seeds) + 2) + '|')
            for ep in EPOCHS:
                cells = []
                for s in seeds:
                    c = blk['eval500'].get(str(s), {}).get(str(ep))
                    cells.append('—' if c is None else
                                 f"{c['success']:.3f} "
                                 f"[{c['wilson95'][0]:.3f}, {c['wilson95'][1]:.3f}]")
                a = blk['aggregate']['per_epoch'].get(str(ep))
                agg_cell = ('—' if a is None
                            else f"{a['mean']:.3f} +/- {a['std']:.3f} (n={a['n_seeds']})")
                L.append(f'| {ep // 1000}k | ' + ' | '.join(cells) + f' | {agg_cell} |')
            L.append('')
        L.append('### late-checkpoint pooled mean (300k-500k, checkpoint x seed)')
        L.append('')
        L.append('| gamma | n | mean | std | 95% CI half-width | min | max |')
        L.append('|---|---|---|---|---|---|---|')
        for gamma, _ in ARMS:
            b = data[env][gamma]['aggregate']['late_300k_500k']
            if b:
                L.append(f"| {gamma} | {b['n_measurements']} | {b['mean']:.3f} | "
                         f"{b['std']:.3f} | +/-{b['ci95_half_width']:.3f} | "
                         f"{b['min']:.3f} | {b['max']:.3f} |")
            else:
                L.append(f'| {gamma} | — | — | — | — | — | — |')
        L.append(f"| BC control | — | {meta['bc']:.3f} | — | — | — | — |")
        L.append('')
        L.append('### oscillation (per seed, over the checkpoints present)')
        L.append('')
        L.append('| gamma | seed | min | max | range | largest adjacent-50k jump |')
        L.append('|---|---|---|---|---|---|')
        for gamma, _ in ARMS:
            osc = data[env][gamma]['aggregate']['oscillation']
            for s in seeds:
                o = osc.get(str(s))
                if o is None:
                    L.append(f'| {gamma} | {s} | — | — | — | — |')
                else:
                    L.append(f"| {gamma} | {s} | {o['min']:.3f} | {o['max']:.3f} | "
                             f"{o['range']:.3f} | {o['max_adjacent_jump']:.3f} |")
        L.append('')
    L.append('The pooled rows treat each (checkpoint, seed) 500-episode measurement as one')
    L.append('draw and the interval is a t-interval over those draws, so it describes the')
    L.append('spread over checkpoints as well as over seeds. This arm swings by +/-0.3')
    L.append('between adjacent 50k checkpoints of one run, so no single checkpoint of it')
    L.append('may be quoted on its own.')
    L.append('')
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, 'w') as f:
        f.write('\n'.join(L))
    print(f'wrote {out_md}')


def plot(data, seeds, out_png):
    figstyle.use_style()
    n_env = len(ENVS)
    fig, axes = plt.subplots(1, n_env + 1, figsize=(4.0 * (n_env + 1), 3.2))
    for j, (env, meta) in enumerate(ENVS.items()):
        ax = axes[j]
        for gamma, arm in ARMS:
            agg = data[env][gamma]['aggregate']['per_epoch']
            xs = [ep for ep in EPOCHS if str(ep) in agg]
            if not xs:
                continue
            ys = [agg[str(ep)]['mean'] for ep in xs]
            hw = [agg[str(ep)]['ci95_half_width'] for ep in xs]
            c = figstyle.PALETTE[arm['colour']]
            ax.plot([x / 1000 for x in xs], ys, marker='o', color=c, label=arm['label'])
            ax.fill_between([x / 1000 for x in xs],
                            [max(0.0, y - h) for y, h in zip(ys, hw)],
                            [min(1.0, y + h) for y, h in zip(ys, hw)],
                            color=c, alpha=0.18, linewidth=0)
        ax.axhline(meta['bc'], color=figstyle.INK_MUTED, ls='--', lw=1.0)
        ax.text(495, meta['bc'] + 0.018, f'BC control {meta["bc"]:.3f}',
                color=figstyle.INK_MUTED, fontsize=6, ha='right')
        ax.set_xlabel('checkpoint (k steps)')
        ax.set_ylabel('success (500 episodes)')
        ax.set_title(f'{env}: mean over 3 seeds, 95% band')
        ax.set_ylim(0, meta['ylim'])
        ax.legend()

    # Third panel: the in-loop 50-episode curve, which is where a gamma=0.995-style late
    # collapse shows up first. It is a MONITOR, never a reported number (95% CI ~ +/-0.115).
    ax = axes[-1]
    for j, (env, _) in enumerate(ENVS.items()):
        for gamma, arm in ARMS:
            for s in seeds:
                ser = data[env][gamma]['series'].get(str(s), {})
                if not (ser.get('eval_steps') and ser.get('eval_success')):
                    continue
                ax.plot([x / 1000 for x in ser['eval_steps']], ser['eval_success'],
                        color=figstyle.PALETTE[arm['colour']],
                        ls='-' if j == 0 else ':', alpha=0.55, lw=1.0,
                        label=f'{env} g={gamma}' if s == seeds[0] else None)
    ax.set_xlabel('step (k)')
    ax.set_ylabel('success (50 episodes, in-loop)')
    ax.set_title('in-loop monitor (noisy: 95% CI ~ +/-0.115)')
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f'wrote {out_png}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', required=True, help='dir of eval500_*.json reports')
    ap.add_argument('--exp', required=True, help='experiment root holding the run groups')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--name', default='2026-09-07-affine-discount-ladder')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--table', default='affine_discount_ladder.md')
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(',')]
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out_dir or os.path.join(repo, 'docs', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    data = aggregate(collect(args.logs, args.exp, seeds), seeds)
    report = {
        'note': ('affine strict at gamma=0.98 vs 0.99 on cube and pointmaze, 3 seeds each; '
                 'the ladder is 500-episode tools/eval_checkpoint.py evals (EVAL_WORKERS=1) '
                 'at every 50k checkpoint. In-loop eval is 50 episodes (+/-0.115) and is a '
                 'collapse monitor only. Pre-registration: '
                 'docs/design/2026-09-07-discount-sweep.md.'),
        'envs': data}
    with open(os.path.join(out_dir, f'{args.name}.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print(f'wrote {os.path.join(out_dir, args.name)}.json')
    plot(data, seeds, os.path.join(out_dir, f'{args.name}.png'))
    write_table(data, seeds, os.path.join(repo, 'docs', 'tables', args.table))


if __name__ == '__main__':
    main()
