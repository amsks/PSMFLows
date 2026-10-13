"""Does critic A act better than critic B over a k-step segment from the same state? (2026-10-13,
test T2 of kisski's mailbox/kisski/2026-10-13T14-29-51Z.md)

The one-action swap of diag_rollout_value_check.py had no power: the continuation recovers from a
single action, so candidates differed by 0.035 in outcome, below the rollout noise. Here a side's
whole policy (best of 64 prior draws scored by its own critic, re-drawn every step) acts for k steps.

Main episodes are acted by critic A (the deployed policy). At each probe step t in +probe_ts:
  1. snapshot the simulator;
  2. openness: +n_open branches of A from the snapshot to the time limit; p_open = their success rate.
     Keep the probe only if +open_lo <= p_open <= +open_hi (otherwise count it as skipped);
  3. on a kept probe, for each k in +ks: +n_rep branches "A acts k steps, then A" and +n_rep branches
     "B acts k steps, then A". Outcome per branch: success (0/1) and gamma^steps-to-success;
  4. restore the snapshot and continue the main episode.
Never one fixed latent for k steps: each step draws fresh prior latents (fixed-noise policies score
about 0.006). A determinism check replays the first openness branch's actions from the snapshot.

Run one (seed pair, task) cell (CPU is fine):
  .venv/bin/python tools/diag_segment_swap.py agent=psmgoal \
    env_name=cube-single-play-singletask-task1-v0 \
    agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \
    agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz \
    agent.coef_source=hgoal_each agent.acting=gpi \
    restore_path=<critic A run> restore_epoch=500000 +restore_path_b=<critic B run> \
    eval_episodes=8 seed=0 report_out=<cell json>

Pool cells (kept probes pooled over tasks per seed, then mean and SE over seeds):
  .venv/bin/python tools/diag_segment_swap.py aggregate <out.json> <cell1.json> ...
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax

PROBE_TS = tuple(range(0, 121, 10))
KS = (20, 50)


def probe_keys(ks):
    out = []
    for k in ks:
        out += [f'a{k}_succ', f'b{k}_succ', f'diff{k}_succ', f'a{k}_disc', f'b{k}_disc', f'diff{k}_disc']
    return out


def aggregate(paths, out_path):
    by_seed = {}
    for p in paths:
        with open(p) as f:
            cell = json.load(f)
        s = by_seed.setdefault(cell['seed_tag'], {'probes': [], 'kept': 0, 'skipped': 0, 'ks': cell['ks']})
        s['probes'] += cell['probes']
        s['kept'] += cell['n_kept']
        s['skipped'] += cell['n_skipped']
    ks = next(iter(by_seed.values()))['ks']
    keys = probe_keys(ks) + ['p_open']
    per_seed = {}
    for tag, s in by_seed.items():
        row = {k: float(np.mean([p[k] for p in s['probes']])) if s['probes'] else float('nan') for k in keys}
        row.update({'n_kept': s['kept'], 'n_skipped': s['skipped']})
        per_seed[tag] = row
    table = {}
    for k in keys + ['n_kept', 'n_skipped']:
        v = np.asarray([per_seed[t][k] for t in per_seed], np.float64)
        table[k] = {'mean': float(np.nanmean(v)),
                    'se': float(np.nanstd(v, ddof=1) / np.sqrt(np.isfinite(v).sum())) if len(v) > 1 else None}
    report = {'n_cells': len(paths), 'n_seeds': len(per_seed), 'per_seed': per_seed, 'table': table}
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    for k, v in table.items():
        print(f'{k:14s} {v["mean"]:.3f}' + (f' ± {v["se"]:.3f}' if v['se'] is not None else ''))


def run(cfg):
    import jax

    from tools.diag_rollout_value_check import SimState, load_pair

    base_seed = int(cfg.seed)
    n_ep = int(cfg.eval_episodes)
    n_open, n_rep = int(cfg.get('n_open', 8)), int(cfg.get('n_rep', 8))
    open_lo, open_hi = float(cfg.get('open_lo', 0.2)), float(cfg.get('open_hi', 0.8))
    probe_ts = tuple(int(t) for t in cfg.get('probe_ts', PROBE_TS))
    ks = tuple(int(k) for k in cfg.get('ks', KS))
    ag, eval_env = load_pair(cfg)
    gamma = float(ag['a'].config['goal_discount'])
    keys = {'k': jax.random.PRNGKey(base_seed)}

    def next_key():
        keys['k'], sub = jax.random.split(keys['k'])
        return sub

    def act(side, obs):
        return np.clip(np.asarray(ag[side].sample_actions(observations=obs, seed=next_key(), temperature=0)), -1, 1)

    sim = SimState(eval_env)
    env = sim.env
    t_max = int(eval_env.spec.max_episode_steps) if eval_env.spec and eval_env.spec.max_episode_steps else 200

    def branch(snap, horizon, seg_side=None, k=0, record=False):
        """Restore; seg_side acts for the first k steps, A afterwards, to success or horizon steps.
        Returns (success, steps, obs list, action list)."""
        ob = sim.restore(snap)
        obs_seq, act_seq = [], []
        for j in range(1, horizon + 1):
            a = act(seg_side if (seg_side is not None and j <= k) else 'a', ob)
            ob, _, _, _, info = env.step(a)
            if record:
                obs_seq.append(np.asarray(ob))
                act_seq.append(a)
            if bool(info['success']):
                return True, j, obs_seq, act_seq
        return False, horizon, obs_seq, act_seq

    probes, n_skipped, ep_success, det_dev = [], 0, [], None
    t0 = time.time()
    eval_env.reset(seed=base_seed)
    for ep in range(n_ep):
        observation, info = eval_env.reset()
        done, step = False, 0
        while not done:
            if step in probe_ts and step < t_max - 1 and not bool(env._success):
                snap = sim.snapshot()
                horizon = t_max - step
                opens = []
                for r in range(n_open):
                    rec = det_dev is None and r == 0
                    ok, _, obs_seq, act_seq = branch(snap, horizon, record=rec)
                    opens.append(float(ok))
                    if rec:
                        sim.restore(snap)
                        det_dev = 0.0
                        for a_r, o_r in zip(act_seq, obs_seq):
                            det_dev = max(det_dev, float(np.max(np.abs(np.asarray(env.step(a_r)[0]) - o_r))))
                        assert det_dev == 0.0, f'simulator restore is not deterministic: {det_dev}'
                p_open = float(np.mean(opens))
                if open_lo <= p_open <= open_hi:
                    m = {'episode': ep, 't': step, 'p_open': p_open}
                    for k in ks:
                        for side in ('a', 'b'):
                            succ, disc = [], []
                            for _ in range(n_rep):
                                ok, j, _, _ = branch(snap, horizon, seg_side=side, k=k)
                                succ.append(float(ok))
                                disc.append(gamma ** j if ok else 0.0)
                            m[f'{side}{k}_succ'], m[f'{side}{k}_disc'] = float(np.mean(succ)), float(np.mean(disc))
                        m[f'diff{k}_succ'] = m[f'a{k}_succ'] - m[f'b{k}_succ']
                        m[f'diff{k}_disc'] = m[f'a{k}_disc'] - m[f'b{k}_disc']
                    probes.append(m)
                    print(f'ep {ep} t {step}: p_open {p_open:.2f} ' + ' '.join(
                        f'k{k} succ A {m[f"a{k}_succ"]:.2f} B {m[f"b{k}_succ"]:.2f}' for k in ks)
                        + f' ({time.time() - t0:.0f} s)', flush=True)
                else:
                    n_skipped += 1
                    print(f'ep {ep} t {step}: p_open {p_open:.2f} skipped ({time.time() - t0:.0f} s)', flush=True)
                observation = sim.restore(snap)
            observation, _, terminated, truncated, info = eval_env.step(act('a', observation))
            done = terminated or truncated
            step += 1
        ep_success.append(bool(info.get('success', False)))
        print(f'episode {ep}: success={ep_success[-1]} length={step}', flush=True)

    summary = {k: float(np.mean([p[k] for p in probes])) if probes else float('nan')
               for k in probe_keys(ks) + ['p_open']}
    report = {
        'env': str(cfg.env_name), 'run_a': str(cfg.restore_path), 'run_b': str(cfg.restore_path_b),
        'restore_epoch': int(cfg.restore_epoch), 'seed': base_seed,
        'seed_tag': os.path.basename(str(cfg.restore_path))[:5],
        'n_open': n_open, 'n_rep': n_rep, 'open_lo': open_lo, 'open_hi': open_hi, 'probe_ts': list(probe_ts),
        'ks': list(ks), 'gamma': gamma, 't_max': t_max, 'determinism_max_dev': det_dev,
        'main_episode_success': [int(x) for x in ep_success], 'n_kept': len(probes), 'n_skipped': n_skipped,
        'probes': probes, 'summary': summary, 'wall_seconds': round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(cfg.report_out)), exist_ok=True)
    with open(cfg.report_out, 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({'n_kept': len(probes), 'n_skipped': n_skipped, **summary}, indent=2))
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
