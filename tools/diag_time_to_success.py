"""Time to success per acting rule: do the best-of-N actors reach the goal later but more often?
(2026-10-15)

One psmgoal checkpoint (critic A, restore_path), optionally a second (critic B, +restore_path_b), on
the five cube tasks. Readouts, each a set of agent keys forced on top of the run's own flags.json
(tools/eval_checkpoint.py merge_run_config):

  bon64    best of 64 prior draws under critic A, the deployed rule (coef_source=hgoal_each,
           acting=gpi, gpi_num_u=64)
  bon8     the same with gpi_num_u=8
  bc       gpi_num_u=1: the argmax over one prior draw is a behaviour-cloning action through the flow
  actor    the goal actor acting alone (acting=distill, eval_redistill=false, eval_goal_source=relabel),
           the run's own in-loop actor readout (actor_rel in scripts/campaign_2026-10/ja_eval_watcher_cube.sh)
  bon64_B  bon64 under critic B (added when +restore_path_b is given)

Per task the relabel batch is drawn once and shared by every readout; each readout infers its goal
set on it (infer_eval_goals with the task reward, shifted by eval_reward_shift) and plays
eval_episodes episodes with utils.evaluation.evaluate at the same eval seed, so all readouts of a
task see the same reset states. Per episode: success (max of info['success'] over the steps), the
success flag at the final step, the first step at which success became 1 (None if never) and the
episode length. The cube env terminates at the goal, so for a successful episode the success step
equals the episode length and the final-step flag equals the max.

Per readout and task, and five-task pooled: success rate with its standard error, mean and median
time to success over the SUCCESSFUL episodes, mean episode length, and the survival curve
S(t) = fraction of episodes that have succeeded by step t for t = 0..horizon (reliability is S at
the horizon, speed is where S rises). Everything, per-episode records included, goes to JSON
(report_out).

Run (GPU):
  .venv/bin/python tools/diag_time_to_success.py agent=psmgoal env_name=cube-single-play \\
    agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \\
    agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz agent.use_point_preimage=true \\
    restore_path=<critic A run> restore_epoch=500000 +restore_path_b=<critic B run> \\
    +readouts="bon64 bon8 bc actor" +tasks="1 2 3 4 5" eval_episodes=100 seed=0 \\
    report_out=$PSM_DATA/logs/time_to_success/<tag>.json

Plot the survival curves of a report (pooled, and one panel per task) to PNG:
  .venv/bin/python tools/diag_time_to_success.py plot <report.json> [<out dir>]
  (default out dir $PSM_DATA/logs/figures)

The same curves from tools/diag_failure_stages.py cells (critic A vs B, per-episode success and
length; the length there counts the reset state, so time to success is length - 1):
  .venv/bin/python tools/diag_time_to_success.py from_failure_stages <out.json> <cell.json> ...
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax (imported by the agents in run())

READOUTS = {
    'bon64': {'coef_source': 'hgoal_each', 'acting': 'gpi', 'gpi_num_u': 64},
    'bon8': {'coef_source': 'hgoal_each', 'acting': 'gpi', 'gpi_num_u': 8},
    'bc': {'coef_source': 'hgoal_each', 'acting': 'gpi', 'gpi_num_u': 1},
    'actor': {'acting': 'distill', 'eval_redistill': False, 'eval_goal_source': 'relabel'},
    'bon64_B': {'coef_source': 'hgoal_each', 'acting': 'gpi', 'gpi_num_u': 64},
}
#: readouts evaluated on critic B (restore_path_b) instead of critic A
B_READOUTS = ('bon64_B',)
DEFAULT_READOUTS = 'bon64 bon8 bc actor'
ENV_PREFIX_DEFAULT = 'cube-single-play'
HORIZON = 200          # cube-single max_episode_steps
MARKS = (50, 100, 150)


# ----------------------------------------------------------------------------- pure helpers
def first_success_step(flags):
    """The first step (1-based: the flag read after the k-th env step is step k) at which the
    success flag is nonzero, or None if it never is."""
    flags = np.asarray(flags, np.float64).reshape(-1)
    hit = np.flatnonzero(flags > 0)
    return int(hit[0]) + 1 if hit.size else None


def survival_curve(t_success, horizon=HORIZON):
    """S(t) for t = 0..horizon: the fraction of episodes whose success step is <= t. Episodes with
    t_success None never contribute. S(0) is 0 (no episode succeeds before its first step)."""
    t = np.asarray([horizon + 1 if v is None else int(v) for v in t_success], np.int64)
    grid = np.arange(horizon + 1)
    if t.size == 0:
        return np.zeros(horizon + 1, np.float64)
    return (t[None, :] <= grid[:, None]).mean(axis=1)


def episode_record(traj):
    """Per-episode readouts from one utils.evaluation.evaluate trajectory."""
    infos = traj['info']
    flags = [float(np.max(np.asarray(info.get('success', 0.0)))) for info in infos]
    return {
        'success': float(max(flags) if flags else 0.0),
        'success_final': float(flags[-1]) if flags else 0.0,
        't_success': first_success_step(flags),
        'length': len(infos),
    }


def mean_se(x):
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return {'mean': None, 'se': None, 'n': 0}
    se = float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0
    return {'mean': float(x.mean()), 'se': se, 'n': int(x.size)}


def summarise(records, horizon=HORIZON, marks=MARKS):
    """Success rate (with SE), time to success over the successful episodes, episode length and
    the survival curve of one set of episode records."""
    succ = [r for r in records if r['t_success'] is not None]
    ts = [r['t_success'] for r in succ]
    curve = survival_curve([r['t_success'] for r in records], horizon)
    return {
        'n_episodes': len(records),
        'success': mean_se([r['success'] for r in records]),
        'success_final': mean_se([r['success_final'] for r in records]),
        't_success_mean': float(np.mean(ts)) if ts else None,
        't_success_median': float(np.median(ts)) if ts else None,
        't_success_se': mean_se(ts)['se'] if ts else None,
        'n_success': len(succ),
        'length_mean': float(np.mean([r['length'] for r in records])) if records else None,
        'by_step': {str(m): float(curve[min(m, horizon)]) for m in marks},
        'survival': curve.tolist(),
    }


def fmt_row(name, s):
    """One table line: readout | success ± se | median t | mean t | S(50) S(100) S(150) | mean length."""
    def f(v, w=6, p=1):
        return f'{v:{w}.{p}f}' if v is not None else ' ' * (w - 4) + 'none'
    by = ' '.join(f'{s["by_step"][str(m)]:.3f}' for m in MARKS)
    return (f'{name:10s} success {s["success"]["mean"]:.3f} ± {s["success"]["se"]:.3f}  '
            f'median_t {f(s["t_success_median"])}  mean_t {f(s["t_success_mean"])}  '
            f'by 50/100/150 {by}  length {f(s["length_mean"])}  n {s["n_episodes"]}')


def parse_names(v, default):
    if v is None:
        v = default
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return str(v).strip('[]()').replace(',', ' ').split()


def parse_tasks(v):
    return [int(t) for t in parse_names(v, '1 2 3 4 5')]


def task_env_name(prefix, task):
    """The singletask env id of one task, as scripts/eval500.sh forms it."""
    return f'{prefix}-singletask-task{int(task)}-v0'


# ----------------------------------------------------------------------------- failure-stage cells
def records_from_failure_stages(cell):
    """Episode records of a tools/diag_failure_stages.py cell, {'a': [...], 'b': [...]}. That tool
    records the reset state as the first row, so an episode of L env steps has length L + 1; the env
    terminates at the goal, so a successful episode's success step is L."""
    out = {}
    for side in ('a', 'b'):
        recs = []
        for e in cell['episodes'][side]:
            length = int(e['stats']['length']) - 1
            ok = bool(e['success'])
            recs.append({'success': float(ok), 'success_final': float(ok),
                         't_success': length if ok else None, 'length': length})
        out[side] = recs
    return out


def from_failure_stages(out_path, cell_paths, horizon=HORIZON):
    """Pool failure-stage cells (one per seed pair x task) into per-task and overall curves for
    critic A and B. The A/B label is read from each cell's run_a / run_b."""
    per_task, pooled, meta = {}, {'a': [], 'b': []}, {}
    for p in cell_paths:
        with open(p) as fh:
            cell = json.load(fh)
        recs = records_from_failure_stages(cell)
        task = str(cell.get('env', p))
        meta.setdefault('run_a', cell.get('run_a'))
        meta.setdefault('run_b', cell.get('run_b'))
        for side in ('a', 'b'):
            per_task.setdefault(task, {}).setdefault(side, []).extend(recs[side])
            pooled[side].extend(recs[side])
    report = {
        'tool': 'diag_time_to_success', 'source': 'diag_failure_stages', 'cells': [str(p) for p in cell_paths],
        'horizon': horizon, **meta,
        'per_task': {t: {s: summarise(r, horizon) for s, r in sides.items()} for t, sides in per_task.items()},
        'pooled': {s: summarise(r, horizon) for s, r in pooled.items()},
        'per_episode': {t: sides for t, sides in per_task.items()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    with open(out_path, 'w') as fh:
        json.dump(report, fh, indent=1)
    for t in sorted(per_task):
        for s in ('a', 'b'):
            print(f'{t[-14:]:14s} ' + fmt_row(s, report['per_task'][t][s]))
    for s in ('a', 'b'):
        print('pooled         ' + fmt_row(s, report['pooled'][s]))
    print(f'report -> {out_path}')
    return report


# ----------------------------------------------------------------------------- plotting
def plot(report_path, out_dir=None):
    """Survival curves of one report: pooled (one line per readout) and one panel per task."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with open(report_path) as fh:
        rep = json.load(fh)
    out_dir = out_dir or os.path.join(os.environ.get('PSM_DATA', '.'), 'logs', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(report_path))[0]
    names = list(rep['pooled'].keys())
    tasks = sorted(rep['per_task'].keys())

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in names:
        s = rep['pooled'][name]
        ax.plot(s['survival'], label=f'{name} ({s["success"]["mean"]:.2f})')
    ax.set_xlabel('step t')
    ax.set_ylabel('fraction succeeded by t')
    ax.set_ylim(0, 1)
    ax.set_title(f'{stem}: pooled over tasks')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    pooled_png = os.path.join(out_dir, f'{stem}_survival_pooled.png')
    fig.tight_layout()
    fig.savefig(pooled_png, dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(tasks), figsize=(3.2 * len(tasks), 3.4), sharey=True, squeeze=False)
    for ax, t in zip(axes[0], tasks):
        for name in names:
            s = rep['per_task'][t].get(name)
            if s is None:
                continue
            ax.plot(s['survival'], label=f'{name} ({s["success"]["mean"]:.2f})')
        ax.set_title(f'task {t}')
        ax.set_xlabel('step t')
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel('fraction succeeded by t')
    fig.tight_layout()
    task_png = os.path.join(out_dir, f'{stem}_survival_per_task.png')
    fig.savefig(task_png, dpi=130)
    plt.close(fig)
    print(f'figures -> {pooled_png}\n           {task_png}')
    return pooled_png, task_png


# ----------------------------------------------------------------------------- rollouts
def build_dataset(train_dataset, config):
    """The relabel dataset with the preimage arrays spliced in, as tools/eval_checkpoint.py."""
    from utils.datasets import Dataset
    from utils.flow_inversion import load_augmented_dataset, repair_invalid_preimages

    train_dataset = dict(train_dataset)
    if config.get('preimage_path'):
        aug, _ = repair_invalid_preimages(load_augmented_dataset(config['preimage_path']))
        assert aug['observations'].shape[0] == train_dataset['observations'].shape[0], \
            'preimage npz row count != eval dataset (wrong env or stale npz)'
        for k in aug:
            if k.startswith('noise_preimage'):
                train_dataset[k] = aug[k]
    ds = Dataset.create(**train_dataset)
    if config.get('preimage_path'):
        ds.return_preimage_noise = True
        ds.preimage_point_mode = bool(config.get('use_point_preimage', False))
    return ds


def run(cfg):
    import ml_collections
    from omegaconf import OmegaConf

    from agents import agents
    from envs.env_utils import make_env_and_datasets
    from main import _lists_to_tuples
    from tools.eval_checkpoint import _cli_agent_keys, merge_run_config
    from utils.evaluation import evaluate
    from utils.flax_utils import restore_agent
    from utils.log_utils import write_report

    t_start = time.time()
    readouts = parse_names(cfg.get('readouts', None), DEFAULT_READOUTS)
    run_b = cfg.get('restore_path_b', None)
    if run_b and not any(r in B_READOUTS for r in readouts):
        readouts.append('bon64_B')
    for r in readouts:
        assert r in READOUTS, f'readout {r!r} must be one of {sorted(READOUTS)}'
        assert run_b or r not in B_READOUTS, f'readout {r} needs +restore_path_b'
    tasks = parse_tasks(cfg.get('tasks', None))
    horizon = int(cfg.get('horizon', HORIZON))
    prefix = str(cfg.env_name) if cfg.env_name else ENV_PREFIX_DEFAULT
    assert '-singletask' not in prefix, 'env_name is the dataset prefix (cube-single-play); tasks come from +tasks'
    assert cfg.restore_path is not None, 'needs a trained checkpoint (restore_path)'
    base_seed, n_ep = int(cfg.seed), int(cfg.eval_episodes)
    shift = float(cfg.get('eval_reward_shift', 1.0))

    cli_agent = OmegaConf.to_container(cfg.agent, resolve=True)
    configs, prov, forced = {}, {}, {}
    for name in readouts:
        rp = str(run_b) if name in B_READOUTS else str(cfg.restore_path)
        merged, pv = merge_run_config(cli_agent, rp, _cli_agent_keys())
        assert merged.get('agent_name') == 'psmgoal', 'diag_time_to_success evaluates psmgoal checkpoints'
        forced[name] = dict(READOUTS[name])
        merged.update(forced[name])
        configs[name] = (ml_collections.ConfigDict(_lists_to_tuples(merged)), rp, merged)
        prov[name] = pv
    print(f'=== diag_time_to_success: readouts={readouts} tasks={tasks} episodes/task={n_ep} seed={base_seed}')
    print(f'    restore={cfg.restore_path} restore_b={run_b} epoch={cfg.restore_epoch} horizon={horizon}')
    for name in readouts:
        c = configs[name][0]
        print(f'    {name:8s} <- {configs[name][1]}: forced {forced[name]}; coef_source={c["coef_source"]} '
              f'acting={c["acting"]} gpi_num_u={int(c["gpi_num_u"])} k_goals={int(c["k_goals"])} '
              f'gpi_hold={int(c.get("gpi_hold", 1))} eval_redistill={c.get("eval_redistill")}', flush=True)

    np.random.seed(base_seed)
    agents_by = {}
    per_task, episodes = {name: {} for name in readouts}, {name: {} for name in readouts}
    for t in tasks:
        env_name = task_env_name(prefix, t)
        _, eval_env, train_dataset, _ = make_env_and_datasets(env_name, frame_stack=cfg.frame_stack)
        ds = build_dataset(train_dataset, configs[readouts[0]][0])
        if not agents_by:
            ex = ds.sample(1)
            for name in readouts:
                config, rp, _ = configs[name]
                a = agents['psmgoal'].create(base_seed, ex['observations'], ex['actions'], config)
                agents_by[name] = restore_agent(a, rp, cfg.restore_epoch)
        zb = ds.sample(min(ds.size, int(cfg.get('eval_relabel_size', 10000))))
        rew = np.asarray(zb['rewards'], np.float32) + shift
        for name in readouts:
            t0 = time.time()
            ag = agents_by[name].infer_eval_goals(zb, rew)
            t_inf = time.time() - t0
            _, trajs, _ = evaluate(agent=ag, env=eval_env, config=configs[name][0], num_eval_episodes=n_ep,
                                   num_video_episodes=0, seed=base_seed)
            recs = [episode_record(tr) for tr in trajs]
            s = summarise(recs, horizon)
            s.update({'env_name': env_name, 'infer_seconds': round(t_inf, 1), 'seconds': round(time.time() - t0, 1)})
            per_task[name][str(t)] = s
            episodes[name][str(t)] = recs
            print(f'task {t} ' + fmt_row(name, s) + f'  ({s["seconds"]}s)', flush=True)

    pooled = {name: summarise([r for t in tasks for r in episodes[name][str(t)]], horizon) for name in readouts}
    print('--- five-task pooled (all episodes of all tasks)')
    for name in readouts:
        print(fmt_row(name, pooled[name]), flush=True)

    report = {
        'tool': 'diag_time_to_success',
        'env_prefix': prefix,
        'tasks': tasks,
        'readouts': readouts,
        'readout_forced_keys': forced,
        'readout_restore_path': {name: configs[name][1] for name in readouts},
        'restore_path': str(cfg.restore_path),
        'restore_path_b': None if run_b is None else str(run_b),
        'restore_epoch': int(cfg.restore_epoch),
        'seed': base_seed,
        'train_seed': {name: prov[name].get('train_seed') for name in readouts},
        'episodes_per_task': n_ep,
        'horizon': horizon,
        'eval_reward_shift': shift,
        'per_task': {str(t): {name: per_task[name][str(t)] for name in readouts} for t in tasks},
        'pooled': pooled,
        'per_episode': {name: episodes[name] for name in readouts},
        'agent_config': {name: json.loads(json.dumps(configs[name][2], default=str)) for name in readouts},
        'config_provenance': prov,
        'wall_seconds': round(time.time() - t_start, 1),
    }
    write_report(report, cfg, 'diag_time_to_success.json')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'plot':
        plot(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif len(sys.argv) > 1 and sys.argv[1] == 'from_failure_stages':
        from_failure_stages(sys.argv[2], sys.argv[3:])
    else:
        import hydra

        @hydra.main(version_base=None, config_path='../configs', config_name='config')
        def _main(cfg):
            run(cfg)

        _main()
