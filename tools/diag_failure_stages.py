"""Where do critic A's and critic B's episodes fail? (2026-10-13, test T1 of kisski's
mailbox/kisski/2026-10-13T14-29-51Z.md)

Critic A (psmgoal_ja_fbc3_pt_cube) acts at 0.805 on cube, critic B (psmgoal_ja_fbcBC_pt_cube, the
actor pinned to BC) at 0.657. Both act as deployed (acting=gpi, coef_source=hgoal_each, best of 64
prior draws). Each side plays the same eval_episodes reset seeds. Per step the tool records the cube
position, the effector (pinch site) position, the gripper opening, the right pad contact and the
gripper command (action[4]), and labels each episode:

  success           the task's success flag fired
  never_grasped     no step with a grasp
  dropped           grasped at some step, not holding the cube at the time limit
  held_not_placed   holding the cube at the time limit

A grasp at a step: right pad contact >= +contact_min, effector within +grasp_dist of the cube centre
and the gripper opening <= +open_max. A lift: the cube's z above its reset z by +lift_dz. The per-step
arrays are persisted, so the labels can be recomputed with other thresholds without new rollouts.

Run one (seed pair, task) cell (CPU is fine):
  .venv/bin/python tools/diag_failure_stages.py agent=psmgoal \
    env_name=cube-single-play-singletask-task1-v0 \
    agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \
    agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz \
    agent.coef_source=hgoal_each agent.acting=gpi \
    restore_path=<critic A run> restore_epoch=500000 +restore_path_b=<critic B run> \
    eval_episodes=20 seed=0 report_out=<cell json>

Pool cells (mean over tasks per seed, then mean and SE over seeds):
  .venv/bin/python tools/diag_failure_stages.py aggregate <out.json> <cell1.json> ...
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax

LABELS = ('success', 'never_grasped', 'dropped', 'held_not_placed')
THRESHOLDS = {'contact_min': 0.2, 'grasp_dist': 0.03, 'open_max': 0.9, 'lift_dz': 0.02}


def label_episode(steps, success, th):
    """steps: dict of per-step arrays (cube, eff, opening, contact). Returns (label, stats)."""
    cube, eff = np.asarray(steps['cube']), np.asarray(steps['eff'])
    opening, contact = np.asarray(steps['opening']), np.asarray(steps['contact'])
    dist = np.linalg.norm(eff - cube, axis=1)
    grasp = (contact >= th['contact_min']) & (dist <= th['grasp_dist']) & (opening <= th['open_max'])
    lifted = cube[:, 2] > cube[0, 2] + th['lift_dz']
    first = lambda m: int(np.argmax(m)) if m.any() else None  # noqa: E731
    stats = {
        'length': int(len(dist)),
        'first_grasp': first(grasp),
        'first_lift': first(lifted & grasp),
        'grasp_steps': int(grasp.sum()),
        # separate grasp events: the number of off -> on switches
        'grasp_events': int(grasp[0]) + int((grasp[1:] & ~grasp[:-1]).sum()),
        'ever_lifted': bool((lifted & grasp).any()),
        'min_dist': float(dist.min()),
    }
    if success:
        label = 'success'
    elif not grasp.any():
        label = 'never_grasped'
    elif grasp[-1]:
        label = 'held_not_placed'
    else:
        label = 'dropped'
    return label, stats


def side_summary(episodes):
    n = len(episodes)
    out = {f'share_{lab}': sum(e['label'] == lab for e in episodes) / n for lab in LABELS}
    fg = [e['stats']['first_grasp'] for e in episodes if e['stats']['first_grasp'] is not None]
    fl = [e['stats']['first_lift'] for e in episodes if e['stats']['first_lift'] is not None]
    succ = [e for e in episodes if e['label'] == 'success']
    out.update({
        'n_episodes': n,
        'first_grasp_mean': float(np.mean(fg)) if fg else float('nan'),
        'first_lift_mean': float(np.mean(fl)) if fl else float('nan'),
        'grasp_events_mean': float(np.mean([e['stats']['grasp_events'] for e in episodes])),
        'success_length_mean': float(np.mean([e['stats']['length'] for e in succ])) if succ else float('nan'),
        # sanity of the grasp detector: successes it saw no grasp in
        'success_without_grasp': sum(e['stats']['first_grasp'] is None for e in succ) / max(len(succ), 1),
    })
    return out


def relabel(cell, th):
    for side in ('a', 'b'):
        for e in cell['episodes'][side]:
            e['label'], e['stats'] = label_episode(e['steps'], e['success'], th)
        cell['summary'][side] = side_summary(cell['episodes'][side])
    return cell


def aggregate(paths, out_path, th=None):
    """Mean over tasks per seed, then mean and SE over seeds, per side. th relabels from the steps."""
    by_seed = {}
    for p in paths:
        with open(p) as f:
            cell = json.load(f)
        if th is not None:
            cell = relabel(cell, th)
        by_seed.setdefault(cell['seed_tag'], []).append(cell['summary'])
    keys = [f'share_{lab}' for lab in LABELS] + ['first_grasp_mean', 'first_lift_mean', 'grasp_events_mean',
                                                 'success_length_mean', 'success_without_grasp']
    table = {}
    for side in ('a', 'b'):
        per_seed = {s: {k: float(np.nanmean([c[side][k] for c in cs])) for k in keys} for s, cs in by_seed.items()}
        table[side] = {}
        for k in keys:
            v = np.asarray([per_seed[s][k] for s in per_seed])
            table[side][k] = {'mean': float(v.mean()),
                              'se': float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None}
        table[side]['per_seed'] = per_seed
    report = {'n_cells': len(paths), 'n_seeds': len(by_seed), 'thresholds': th or 'as recorded', 'table': table}
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    for k in keys:
        row = [f'{table[s][k]["mean"]:.3f}' + (f' ± {table[s][k]["se"]:.3f}' if table[s][k]['se'] is not None
                                               else '') for s in ('a', 'b')]
        print(f'{k:24s} A {row[0]:16s} B {row[1]}')


def run(cfg):
    import jax

    from tools.diag_rollout_value_check import load_pair

    base_seed = int(cfg.seed)
    n_ep = int(cfg.eval_episodes)
    th = {k: float(cfg.get(k, v)) for k, v in THRESHOLDS.items()}
    ag, eval_env = load_pair(cfg)
    keys = {'k': jax.random.PRNGKey(base_seed)}

    def next_key():
        keys['k'], sub = jax.random.split(keys['k'])
        return sub

    env = eval_env.unwrapped

    def record(steps, info, action):
        # read the simulator directly: the wrapped env's reset info does not carry these keys
        info = env.compute_ob_info()
        steps['cube'].append(np.round(np.asarray(info['privileged/block_0_pos'], np.float64), 5).tolist())
        steps['eff'].append(np.round(np.asarray(info['proprio/effector_pos'], np.float64), 5).tolist())
        steps['opening'].append(round(float(info['proprio/gripper_opening'][0]), 5))
        steps['contact'].append(round(float(info['proprio/gripper_contact'][0]), 5))
        steps['cmd'].append(None if action is None else round(float(action[4]), 5))

    t0 = time.time()
    episodes = {}
    for side in ('a', 'b'):
        policy = ag[side]
        eval_env.reset(seed=base_seed)          # both sides see the same sequence of reset states
        episodes[side] = []
        for ep in range(n_ep):
            observation, info = eval_env.reset()
            steps = {'cube': [], 'eff': [], 'opening': [], 'contact': [], 'cmd': []}
            record(steps, info, None)
            done = False
            while not done:
                action = np.clip(np.asarray(policy.sample_actions(observations=observation, seed=next_key(),
                                                                  temperature=0)), -1, 1)
                observation, _, terminated, truncated, info = eval_env.step(action)
                steps['cmd'][-1] = round(float(action[4]), 5)   # the command issued at the previous state
                record(steps, info, None)
                done = terminated or truncated
            success = bool(info.get('success', False))
            label, stats = label_episode(steps, success, th)
            episodes[side].append({'episode': ep, 'success': success, 'label': label, 'stats': stats,
                                   'steps': steps})
            print(f'{side} ep {ep}: {label} length={stats["length"]} first_grasp={stats["first_grasp"]} '
                  f'events={stats["grasp_events"]} min_dist={stats["min_dist"]:.3f} '
                  f'({time.time() - t0:.0f} s)', flush=True)

    report = {
        'env': str(cfg.env_name), 'run_a': str(cfg.restore_path), 'run_b': str(cfg.restore_path_b),
        'restore_epoch': int(cfg.restore_epoch), 'seed': base_seed,
        'seed_tag': os.path.basename(str(cfg.restore_path))[:5], 'thresholds': th,
        'summary': {side: side_summary(episodes[side]) for side in ('a', 'b')},
        'episodes': episodes, 'wall_seconds': round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(cfg.report_out)), exist_ok=True)
    with open(cfg.report_out, 'w') as f:
        json.dump(report, f)
    print(json.dumps(report['summary'], indent=2))
    print('report ->', cfg.report_out)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'aggregate':
        aggregate(sys.argv[3:], sys.argv[2])
    else:
        import hydra

        @hydra.main(version_base=None, config_path='../configs', config_name='config')
        def _main(cfg):
            run(cfg)

        _main()
