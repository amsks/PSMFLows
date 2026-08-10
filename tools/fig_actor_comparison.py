"""Figure F3 -- which way of choosing a latent actually earns its keep, on cube-single.

(a) 500-episode success with Wilson intervals, in increasing order of what the method is
    allowed to do:
      1. behaviour cloning        -- the frozen flow, a fresh prior latent every step
      2. flow-GPI over fixed u    -- selection over the fixed-latent family (Rung 1)
      3. LatentFlowPSM, acting=gpi-- per-step latent argmax, actor-free ablation
      4. LatentFlowPSM, acting=actor -- the amortised latent actor
      5. FQL                      -- per-task, sees the reward at training time; a
                                     ceiling, not a peer, and drawn as such
    The family-ceiling line marks what selection over fixed latents could ever reach.
(b) Training curves, mean +/- 95% CI across seeds, with the BC control as a flat line.

Every bar reads a JSON written by tools/eval_checkpoint.py; bars whose JSON is missing
are skipped and named in the report, so a partial run is obvious rather than silent.

Run: .venv/bin/python tools/fig_actor_comparison.py
"""
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.figstyle import (INK, INK_MUTED, LOG_DIR, PALETTE, mean_ci, save,
                            use_style, wilson)

import matplotlib.pyplot as plt

EXP = '/data-local/amsks/PSMFLows/exp/PSMFLows'
PSMFLOW_GROUP = f'{EXP}/psmflow_latentpsm_cube_021456'
FQL_GROUP_GLOB = f'{EXP}/fqlbaseline_cube_a300_*'
RUNG1_RUN = f'{EXP}/audit_psmflow_cube_204354'
SEEDS = [0, 1, 2, 3, 4]


def load_eval(name):
    p = f'{LOG_DIR}/{name}.json'
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def pooled(reports):
    """Pool per-episode outcomes across seeds -> one Wilson interval.

    Pooling is the right summary for a bar: it is the success rate of the method over
    all episodes run. The per-seed spread is drawn on top as dots, and the across-seed
    mean +/- CI goes in the report, so neither view is hidden.
    """
    k = sum(r['num_success'] for r in reports)
    n = sum(r['num_episodes'] for r in reports)
    p, lo, hi = wilson(k, n)
    per_seed = [r['success'] for r in reports]
    m, half = mean_ci(per_seed)
    return {'success': p, 'ci': [lo, hi], 'num_success': k, 'num_episodes': n,
            'per_seed': per_seed, 'seed_mean': m, 'seed_ci_halfwidth': half}


def eval_curve(run_dir):
    """(steps, success) from a run's eval.csv."""
    p = os.path.join(run_dir, 'eval.csv')
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    key = next((c for c in rows[0] if 'success' in c.lower()), None)
    if key is None:
        return None
    return ([int(float(r['step'])) for r in rows],
            [float(r[key]) for r in rows])


def main():
    use_style()
    report = {'inputs': {}, 'bars': {}, 'curves': {}, 'missing': []}

    bars = []

    bc = load_eval('eval500_bcflow_cube')
    if bc:
        p, lo, hi = wilson(bc['num_success'], bc['num_episodes'])
        bars.append(dict(key='bc', label='BC', value=p, ci=(lo, hi),
                         color=PALETTE[5], per_seed=[], group='fixed'))
        report['bars']['bc'] = {'success': p, 'ci': [lo, hi],
                                'num_episodes': bc['num_episodes']}
    else:
        report['missing'].append('eval500_bcflow_cube')

    # Rung-1 flow-GPI over the fixed-latent family. Its checkpoint predates the latent-PSM
    # redesign, so there is no 500-episode rerun: the number is the in-loop 50-episode
    # eval, and the figure labels it as such rather than pretending it is comparable.
    r1 = eval_curve(glob.glob(f'{RUNG1_RUN}/sd*')[0]) if glob.glob(f'{RUNG1_RUN}/sd*') else None
    if r1:
        steps, succ = r1
        val = float(np.mean(succ[-4:]))
        lo, hi = float(np.min(succ[-4:])), float(np.max(succ[-4:]))
        bars.append(dict(key='rung1_gpi', label='flow-GPI\nfixed $u$', value=val,
                         ci=(lo, hi), color=PALETTE[4], per_seed=[], group='fixed',
                         note='50-ep in-loop evals, late window'))
        report['bars']['rung1_gpi'] = {
            'success': val, 'range_last4_evals': [lo, hi], 'episodes_per_eval': 50,
            'caveat': 'in-loop 50-episode evals; pre-redesign checkpoint, no 500-ep rerun',
            'steps': steps, 'curve': succ}
    else:
        report['missing'].append('rung1 eval.csv')

    for key, suffix, label, color in [
            ('latent_gpi', '_gpi', 'LFPSM\ngpi', PALETTE[2]),
            ('latent_actor', '', 'LFPSM\nactor', PALETTE[0])]:
        reps, miss = [], []
        for s in SEEDS:
            r = load_eval(f'eval500_latentpsm_cube_sd{s}{suffix}')
            (reps if r else miss).append(r if r else f'sd{s}{suffix}')
        if reps:
            agg = pooled(reps)
            bars.append(dict(key=key, label=label, value=agg['success'],
                             ci=tuple(agg['ci']), color=color,
                             per_seed=agg['per_seed'], group='latent'))
            report['bars'][key] = agg
        report['missing'] += [f'eval500_latentpsm_cube_{m}' for m in miss]

    fql_reps = [r for r in (load_eval(f'eval500_fql_cube_sd{s}') for s in (0, 1, 2)) if r]
    if fql_reps:
        agg = pooled(fql_reps)
        bars.append(dict(key='fql', label='FQL\nper-task', value=agg['success'],
                         ci=tuple(agg['ci']), color=INK_MUTED,
                         per_seed=agg['per_seed'], group='ceiling'))
        report['bars']['fql'] = agg
    else:
        report['missing'] += [f'eval500_fql_cube_sd{s}' for s in (0, 1, 2)]

    # ------------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(6.6, 3.0))
    gs = fig.add_gridspec(1, 2, wspace=0.26, width_ratios=[1.25, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(len(bars))
    for i, b in enumerate(bars):
        lo, hi = b['ci']
        ax.bar(i, b['value'], 0.66, color=b['color'], edgecolor='white', linewidth=0.8,
               hatch='///' if b['group'] == 'ceiling' else None, zorder=3)
        ax.plot([i, i], [lo, hi], color=INK, linewidth=1.0, zorder=5)
        if b['per_seed']:
            ax.scatter([i] * len(b['per_seed']), b['per_seed'], s=7, color=INK,
                       alpha=0.65, zorder=6, linewidths=0)
        ax.text(i, hi + 0.012, f"{b['value']:.3f}", ha='center', fontsize=5.5, color=INK)

    # The ceiling of the fixed-latent family, measured exhaustively in F1 (cube 2/128).
    ceiling = None
    cp = f'{LOG_DIR}/latent_reachability_cube.json'
    if os.path.exists(cp):
        with open(cp) as f:
            c = json.load(f)
        ceiling = c['num_success_any'] / c['num_candidates']
        n_fixed = sum(1 for b in bars if b['group'] == 'fixed')
        if n_fixed:
            ax.plot([-0.5, n_fixed - 0.5], [ceiling, ceiling], color=INK_MUTED,
                    linestyle='--', linewidth=0.9, zorder=4)
            ax.text(-0.42, ceiling + 0.008, 'family ceiling', fontsize=5,
                    color=INK_MUTED)
        report['bars']['family_ceiling'] = {
            'value': ceiling, 'source': cp,
            'reached': c['num_success_any'], 'candidates': c['num_candidates']}

    if any(b['group'] == 'ceiling' for b in bars):
        div = min(i for i, b in enumerate(bars) if b['group'] == 'ceiling') - 0.5
        ax.axvline(div, color=INK_MUTED, linestyle=':', linewidth=0.8)
        ax.text(div - 0.06, 1.02, 'sees reward at train time $\\rightarrow$',
                fontsize=5, color=INK_MUTED, va='bottom', ha='right')
    ax.set_xticks(xs)
    ax.set_xticklabels([b['label'] for b in bars], fontsize=6)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel('success (500 episodes)')
    ax.set_title('(a) cube-single, Wilson 95% intervals', loc='left', fontsize=7)

    # ------------------------------------------------------------------- (b) curves
    ax = fig.add_subplot(gs[0, 1])
    for label, pattern, color in [
            ('LatentFlowPSM', f'{PSMFLOW_GROUP}/sd*', PALETTE[0]),
            ('FQL (per-task)', f'{FQL_GROUP_GLOB}/sd*', INK_MUTED)]:
        curves = [c for c in (eval_curve(d) for d in sorted(glob.glob(pattern))) if c]
        if not curves:
            report['missing'].append(f'curves: {pattern}')
            continue
        n = min(len(c[0]) for c in curves)
        steps = np.array(curves[0][0][:n])
        vals = np.stack([np.array(c[1][:n]) for c in curves])
        means, halves = zip(*[mean_ci(list(vals[:, j])) for j in range(n)])
        means, halves = np.array(means), np.array(halves)
        ax.plot(steps, means, color=color, label=f'{label} ({len(curves)} seeds)')
        ax.fill_between(steps, means - halves, means + halves, color=color, alpha=0.16,
                        linewidth=0)
        report['curves'][label] = {'steps': steps.tolist(), 'mean': means.tolist(),
                                   'ci_halfwidth': halves.tolist(),
                                   'n_seeds': len(curves)}
    if bc:
        ax.axhline(report['bars']['bc']['success'], color=PALETTE[5], linestyle='--',
                   linewidth=1.0, label='behaviour cloning')
    ax.set_xlabel('gradient steps (thousands)')
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'{v / 1000:.0f}'))
    ax.set_ylabel('success (50 episodes)')
    ax.set_title('(b) training curves, mean $\\pm$ 95% CI', loc='left', fontsize=7)
    ax.legend(fontsize=5.5, loc='upper left', handletextpad=0.4, labelspacing=0.25)

    if report['missing']:
        print('MISSING (bars/curves skipped): ' + ', '.join(report['missing']))
    save(fig, 'fig_actor_comparison', report)


if __name__ == '__main__':
    main()
