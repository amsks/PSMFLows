"""Is the trained psmgoal measure a FULL successor measure or a next-state detector? (2026-10-15)

The measure M(s,u,x;g) = phi(s,u,x)^T h(g) + b(s,u,x) is read as a distribution over a fixed
reference set X of n_ref dataset next states: p(x|s,u;g) = softmax_x(M(s,u,x;g)/measure_temp)
(measure_loss=softmax; under measure_loss=squared M is clipped at 0 and normalized to sum 1).
A full successor measure puts p where the goal policy for g goes later; a next-state detector
puts p on the one x nearest the actual next state s' and nothing beyond it.

Probes: the deployed policy (critic A, coef_source=hgoal_each, acting=gpi, the task's rewarding
goals, as eval builds it) acts `episodes` episodes per task in `tasks`; at the steps `probe_ts`
the simulator is snapshotted (SimState of diag_rollout_value_check). At each probe state s two
goals are set:
  goal=task    one of the task's k_goals rewarding goals. task_goal_pick=top_share (default):
               the goal whose own share exp(M(s,u,g;h(g))/t) / (that + sum over the reference
               columns) is highest at the latent u the deployed policy executes at s, i.e. the
               goal the deployed action is scored best for; task_goal_pick=first: eval_goals[0].
  goal=random  one dataset next state drawn uniformly.
For each (s, g) the agent's goal set is replaced by the single goal g (eval_goals=[g],
eval_goal_w=h(g), eval_ref = `n_ref_cols` uniform dataset next states, the number the run's
batch_size - k_goals fixes), 64 prior latents are drawn and u is the best under M, the deployed
rule for that goal; p(x|s,u;g) is read over X with critic A (and critic B at the same (s,u,g)
and its own h_B(g) when restore_path_b is given).

Rollouts: R times from the snapshot, execute the decoded u once, then follow the goal policy
for g (best-of-64 under h(g), re-drawn every step, no latent held) for T steps or until the env
terminates (the singletask cube env terminates on success). The occupancy over X is
  rho(x) = mean_r sum_t gamma^t 1[nn(X, s_t^(r)) = x] / sum_t gamma^t,
t = 0 at the state after the executed u, gamma = the run's discount, each rollout normalized
over the horizon it actually ran. Nearest neighbour on the full observation standardized per
dimension by the dataset std (nn=full) and on the cube xyz columns 19:22 only (nn=cube_xyz).

Per (s, g) and nn mode: tv = 0.5 sum |p - rho|; Spearman(p, rho) over X and over the support
of rho; p's largest mass and whether that x is nn(X, s'); p_next = p(nn(s')), rho_next =
rho(nn(s')); mass_beyond_next = sum_{x != nn(s')} min(p, rho); entropies; the noise floor
tv(rho over the first R/2 rollouts, rho over the second R/2); tv(uniform, rho); tv(p_B, rho).
Aggregates are mean +- SE over probes per goal setting and per (task, goal setting). The
determinism check of diag_rollout_value_check replays one rollout's actions from its snapshot
and must reproduce the observations exactly.

Run (GPU):
  .venv/bin/python tools/diag_measure_fullness.py agent=psmgoal env_name=cube-single-play \\
    agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \\
    agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz \\
    restore_path=<critic A run> restore_epoch=500000 [+restore_path_b=<critic B run>] \\
    +tasks="1 2 3 4 5" +episodes=2 '+probe_ts=[0,20,40,60,80]' +R=8 +T=100 +n_ref=1024 seed=0 \\
    report_out=$PSM_DATA/logs/measure_fullness/<tag>.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax (imported inside run())
from tools.diag_rollout_value_check import SimState, spearman

GOAL_SETTINGS = ('task', 'random')
PROBE_TS = (0, 20, 40, 60, 80)
FORCED_AGENT_KEYS = {'coef_source': 'hgoal_each', 'acting': 'gpi'}
METRIC_KEYS = ('tv', 'tv_noise', 'tv_uniform', 'tv_b', 'tv_a_b', 'spearman', 'spearman_support', 'p_top1',
               'top1_is_next', 'p_next', 'rho_next', 'mass_beyond_next', 'H_p', 'H_rho', 'rho_support',
               'p_b_next', 'top1_b_is_next')


# ------------------------------------------------------------------ pure functions
def obs_std(observations):
    """Per-dimension std of the dataset observations; 1 where the dimension is constant."""
    s = np.asarray(observations, np.float64).std(axis=0)
    s[s < 1e-8] = 1.0
    return s


def nearest(ref_std, q, std, cols=slice(None)):
    """Index in ref of the nearest reference row for each query row, on the columns `cols` of
    the observation standardized by `std`. ref_std is the reference set already divided by std."""
    q = np.atleast_2d(np.asarray(q, np.float64)) / std
    a, b = ref_std[:, cols], q[:, cols]
    d = (a ** 2).sum(1)[None, :] - 2.0 * b @ a.T + (b ** 2).sum(1)[:, None]
    return np.argmin(d, axis=1)


def occupancy(idx_seqs, n_ref, gamma):
    """Discounted visitation over the reference set. idx_seqs: one int array per rollout, the
    nearest-reference index of each visited state (t = 0 first). Each rollout's weights gamma^t
    are normalized over the steps it ran; the rollouts are averaged. Returns (n_ref,), sums to 1."""
    rho = np.zeros(n_ref, np.float64)
    for idx in idx_seqs:
        idx = np.asarray(idx, np.int64).reshape(-1)
        if idx.size == 0:
            continue
        w = gamma ** np.arange(idx.size, dtype=np.float64)
        rho += np.bincount(idx, weights=w, minlength=n_ref) / w.sum()
    return rho / max(len(idx_seqs), 1)


def measure_to_dist(m, loss, temp):
    """The measure row read as a distribution over the columns. softmax: softmax(m / temp);
    squared: m clipped at 0 and normalized (uniform if every entry is <= 0)."""
    m = np.asarray(m, np.float64).reshape(-1)
    if loss == 'softmax':
        z = m / float(temp)
        z = np.exp(z - z.max())
        return z / z.sum()
    z = np.clip(m, 0.0, None)
    return z / z.sum() if z.sum() > 0 else np.full(m.shape, 1.0 / m.size)


def tv(p, q):
    return float(0.5 * np.abs(np.asarray(p, np.float64) - np.asarray(q, np.float64)).sum())


def entropy(p):
    p = np.asarray(p, np.float64)
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def compare(p, rho, next_idx, p_b=None):
    """Per-(s, g) metrics of the measure's distribution p against the rollout occupancy rho over
    the reference set; next_idx is nn(X, s') for the actual next state after the executed u."""
    p, rho = np.asarray(p, np.float64), np.asarray(rho, np.float64)
    support = rho > 0
    top1 = int(np.argmax(p))
    out = {
        'tv': tv(p, rho),
        'tv_uniform': tv(np.full(p.shape, 1.0 / p.size), rho),
        'spearman': spearman(p, rho),
        'spearman_support': spearman(p[support], rho[support]) if support.sum() >= 2 else float('nan'),
        'p_top1': float(p[top1]), 'p_top1_idx': top1,
        'top1_is_next': float(top1 == int(next_idx)),
        'p_next': float(p[next_idx]), 'rho_next': float(rho[next_idx]),
        'mass_beyond_next': float(np.minimum(p, rho).sum() - min(p[next_idx], rho[next_idx])),
        'H_p': entropy(p), 'H_rho': entropy(rho), 'rho_support': int(support.sum()),
    }
    if p_b is not None:
        p_b = np.asarray(p_b, np.float64)
        out.update({'tv_b': tv(p_b, rho), 'tv_a_b': tv(p, p_b), 'p_b_next': float(p_b[next_idx]),
                    'top1_b_is_next': float(int(np.argmax(p_b)) == int(next_idx))})
    return out


def split_half_tv(idx_seqs, n_ref, gamma):
    """Noise floor: tv between the occupancies of the first and second half of the rollouts."""
    half = len(idx_seqs) // 2
    if half == 0:
        return float('nan')
    return tv(occupancy(idx_seqs[:half], n_ref, gamma), occupancy(idx_seqs[half:2 * half], n_ref, gamma))


def summarise(records, keys=METRIC_KEYS):
    """Mean, SE and count over records of each metric key present (nan-aware)."""
    out = {'n': len(records)}
    for k in keys:
        v = np.asarray([r[k] for r in records if k in r], np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        out[k] = {'mean': float(v.mean()), 'se': float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0,
                  'n': int(v.size)}
    return out


def parse_int_list(v, default):
    if v is None:
        return [int(t) for t in default]
    if isinstance(v, (list, tuple)):
        return [int(t) for t in v]
    try:
        from omegaconf import ListConfig
        if isinstance(v, ListConfig):
            return [int(t) for t in v]
    except ImportError:
        pass
    return [int(t) for t in str(v).strip('[]()').replace(',', ' ').split()]


# ------------------------------------------------------------------ run
def run(cfg):
    import jax
    import jax.numpy as jnp
    import ml_collections
    from omegaconf import OmegaConf

    from agents import agents
    from envs.env_utils import make_env_and_datasets
    from main import _lists_to_tuples
    from tools.eval_checkpoint import _cli_agent_keys, merge_run_config
    from tools.eval_state_reward import CUBE_POS, build_dataset, task_env_name
    from utils.flax_utils import restore_agent
    from utils.log_utils import write_report

    t_start = time.time()
    base_seed = int(cfg.seed)
    tasks = parse_int_list(cfg.get('tasks', None), (1, 2, 3, 4, 5))
    n_ep = int(cfg.get('episodes', 2))
    probe_ts = tuple(parse_int_list(cfg.get('probe_ts', None), PROBE_TS))
    R, T, n_ref = int(cfg.get('R', 8)), int(cfg.get('T', 100)), int(cfg.get('n_ref', 1024))
    task_goal_pick = str(cfg.get('task_goal_pick', 'top_share'))
    assert task_goal_pick in ('top_share', 'first')
    assert R >= 1 and T >= 1 and n_ref >= 2
    prefix = str(cfg.env_name)
    assert '-singletask' not in prefix, 'env_name is the dataset prefix (cube-single-play); tasks come from +tasks'
    assert cfg.restore_path is not None, 'needs a trained checkpoint (restore_path)'
    run_b = cfg.get('restore_path_b', None)
    runs = {'a': str(cfg.restore_path)}
    if run_b:
        runs['b'] = str(run_b)

    cli_agent = OmegaConf.to_container(cfg.agent, resolve=True)
    configs, prov = {}, {}
    for side, rp in runs.items():
        merged, prov[side] = merge_run_config(cli_agent, rp, _cli_agent_keys())
        assert merged.get('agent_name') == 'psmgoal', 'diag_measure_fullness reads psmgoal checkpoints'
        merged.update(FORCED_AGENT_KEYS)
        configs[side] = ml_collections.ConfigDict(_lists_to_tuples(merged))
    c = configs['a']
    gamma, temp = float(c['discount']), float(c['measure_temp'])
    loss = str(c.get('measure_loss', 'squared'))
    K, u_clip = int(c['gpi_num_u']), float(c['u_clip'])
    n_ref_cols = max(int(c['batch_size']) - int(c['k_goals']), 1)
    print(f'=== diag_measure_fullness: tasks={tasks} episodes/task={n_ep} probe_ts={probe_ts} R={R} T={T} '
          f'n_ref={n_ref} task_goal_pick={task_goal_pick} seed={base_seed}')
    print(f'    critics: {runs} epoch={cfg.restore_epoch}; forced {FORCED_AGENT_KEYS}; measure_loss={loss} '
          f'temp={temp} gamma={gamma} gpi_num_u={K} k_goals={int(c["k_goals"])} ref_cols={n_ref_cols}', flush=True)

    # Dataset (first task), preimage splice once, the reference set X and the standardization.
    np.random.seed(base_seed)
    _, eval_env, train_dataset, _ = make_env_and_datasets(task_env_name(prefix, tasks[0]), frame_stack=cfg.frame_stack)
    ds = build_dataset(train_dataset, c)
    aug_keys = {k: ds[k] for k in ds if k.startswith('noise_preimage')}
    next_obs_all = np.asarray(train_dataset['next_observations'], np.float32)
    std = obs_std(train_dataset['observations'])
    rng = np.random.default_rng(base_seed + 7919)
    x_idx = np.sort(rng.choice(next_obs_all.shape[0], size=n_ref, replace=False))
    X = next_obs_all[x_idx]
    X_std = X.astype(np.float64) / std
    X_j = jnp.asarray(X)
    nn_modes = {'full': slice(None), 'cube_xyz': CUBE_POS}

    ex = ds.sample(1)
    base = {}
    for side, rp in runs.items():
        a = agents['psmgoal'].create(base_seed, ex['observations'], ex['actions'], configs[side])
        base[side] = restore_agent(a, rp, cfg.restore_epoch)

    @jax.jit
    def choose(agent, obs, key):
        return agent.select_latent(obs, key)

    @jax.jit
    def decode1(agent, obs, u):
        return agent.decode(obs[None], u[None])[0]

    @jax.jit
    def goal_coef(agent, g):
        return agent.w_star(g[None])[0]

    @jax.jit
    def goal_shares(agent, obs, u):
        """Each eval goal's own share at (s, u): `_hgoal_each_score` with one goal column, (G,)."""
        def one(g, hw):
            return agent._hgoal_each_score(obs[None], u[None], g[None], hw[None], agent.eval_ref)[0]
        return jax.vmap(one)(agent.eval_goals, agent.eval_goal_w)

    @jax.jit
    def measure_row(agent, obs, u, hw):
        """M(s,u,x;g) = phi(s,u,x)^T h(g) + b(s,u,x) over the reference set, (n_ref,)."""
        phi, b = agent.mesh_phi_b(obs[None], u[None], X_j)
        return phi[0] @ hw + b[0]

    keys = {'k': jax.random.PRNGKey(base_seed)}

    def next_key():
        keys['k'], sub = jax.random.split(keys['k'])
        return sub

    def act(agent, obs):
        return np.clip(np.asarray(agent.sample_actions(observations=obs, seed=next_key(), temperature=0)), -1, 1)

    def with_goal(agent, g, hw, refs):
        return agent.replace(eval_goals=g[None], eval_goal_w=hw[None], eval_ref=refs)

    sim = SimState(eval_env)
    env, snapshot, restore = sim.env, sim.snapshot, sim.restore
    t_max = int(eval_env.spec.max_episode_steps) if eval_env.spec and eval_env.spec.max_episode_steps else 200

    def rollout(snap, agent_g, a0, record=False):
        """Restore, execute a0, then act with agent_g for T steps or until the env terminates.
        Returns (obs list from the state after a0, actions, success, terminated)."""
        restore(snap)
        obs_seq, act_seq, a, success, term = [], [], a0, False, False
        for t in range(T):
            ob, _, term, _, info = env.step(a)
            obs_seq.append(np.asarray(ob))
            if record:
                act_seq.append(np.asarray(a))
            success = success or bool(info['success'])
            if term or bool(info['success']):
                break
            a = act(agent_g, ob)
        return obs_seq, act_seq, success, term

    probes, det_dev = [], None
    ep_success = {}
    for ti, task in enumerate(tasks):
        env_name = task_env_name(prefix, task)
        if ti > 0:
            _, eval_env, train_dataset, _ = make_env_and_datasets(env_name, frame_stack=cfg.frame_stack)
            train_dataset = dict(train_dataset)
            train_dataset.update(aug_keys)
            from utils.datasets import Dataset
            ds = Dataset.create(**train_dataset)
            ds.return_preimage_noise = True
            ds.preimage_point_mode = bool(c.get('use_point_preimage', False))
            sim = SimState(eval_env)
            env, snapshot, restore = sim.env, sim.snapshot, sim.restore
        # the task's goal set, inferred as eval does (np.random.seed(seed), relabel batch, shift)
        np.random.seed(base_seed)
        zb = ds.sample(min(ds.size, int(cfg.get('eval_relabel_size', 10000))))
        rew = zb['rewards'] + float(cfg.get('eval_reward_shift', 1.0))
        ag = {side: base[side].infer_eval_goals(zb, rew) for side in runs}
        policy = ag['a']
        task_goals = np.asarray(policy.eval_goals)
        ep_success[str(task)] = []
        t0 = time.time()
        eval_env.reset(seed=base_seed)
        for ep in range(n_ep):
            observation, info = eval_env.reset()
            done, step = False, 0
            while not done:
                if step in probe_ts and step < t_max - 1:
                    snap = snapshot()
                    obs_j = jnp.asarray(observation, jnp.float32)
                    u_dep = choose(policy, obs_j, next_key())
                    shares = np.asarray(goal_shares(policy, obs_j, u_dep), np.float64)
                    j_task = int(np.argmax(shares)) if task_goal_pick == 'top_share' else 0
                    for setting in GOAL_SETTINGS:
                        if setting == 'task':
                            g_src, g = j_task, task_goals[j_task]
                        else:
                            g_src = int(rng.integers(next_obs_all.shape[0]))
                            g = next_obs_all[g_src]
                        g_j = jnp.asarray(g, jnp.float32)
                        refs = jnp.asarray(next_obs_all[rng.choice(next_obs_all.shape[0], size=n_ref_cols,
                                                                    replace=False)])
                        agent_g = {side: with_goal(ag[side], g_j, goal_coef(ag[side], g_j), refs) for side in runs}
                        u = choose(agent_g['a'], obs_j, next_key())
                        a0 = np.clip(np.asarray(decode1(agent_g['a'], obs_j, u)), -1, 1)
                        m = {side: np.asarray(measure_row(agent_g[side], obs_j, u, agent_g[side].eval_goal_w[0]),
                                              np.float32) for side in runs}
                        p = {side: measure_to_dist(m[side], loss, temp) for side in runs}
                        seqs, succ, lens, terms = [], [], [], []
                        for r in range(R):
                            rec = det_dev is None and r == 0
                            obs_seq, act_seq, ok, term = rollout(snap, agent_g['a'], a0, record=rec)
                            seqs.append(obs_seq)
                            succ.append(ok)
                            lens.append(len(obs_seq))
                            terms.append(term)
                            if rec:
                                restore(snap)
                                dev = 0.0
                                for a_r, o_r in zip(act_seq, obs_seq):
                                    ob2 = env.step(a_r)[0]
                                    dev = max(dev, float(np.max(np.abs(np.asarray(ob2) - o_r))))
                                det_dev = dev
                                assert det_dev == 0.0, f'simulator restore is not deterministic: {det_dev}'
                        s_next = seqs[0][0]
                        next_dev = max(float(np.max(np.abs(s[0] - s_next))) for s in seqs)
                        rec = {'task': int(task), 'episode': ep, 't': step, 'goal': setting,
                               'goal_source_index': g_src, 'goal_share_top': float(shares[j_task]),
                               'goal_share_mean': float(shares.mean()),
                               'u_norm': float(np.linalg.norm(np.asarray(u))),
                               'rollout_success': [int(s) for s in succ], 'rollout_length': lens,
                               'rollout_terminated': [int(t) for t in terms],
                               'next_state_dev_across_rollouts': next_dev,
                               'M_mean': float(m['a'].mean()), 'M_max': float(m['a'].max()),
                               'M_min': float(m['a'].min()), 'metrics': {}}
                        for nn_name, cols in nn_modes.items():
                            idx_seqs = [nearest(X_std, np.stack(s), std, cols) for s in seqs]
                            rho = occupancy(idx_seqs, n_ref, gamma)
                            next_idx = int(nearest(X_std, s_next, std, cols)[0])
                            met = compare(p['a'], rho, next_idx, p.get('b'))
                            met['tv_noise'] = split_half_tv(idx_seqs, n_ref, gamma)
                            met['next_idx'] = next_idx
                            rec['metrics'][nn_name] = met
                        probes.append(rec)
                        mf = rec['metrics']['full']
                        print(f'task {task} ep {ep} t {step} goal={setting:6s}: tv {mf["tv"]:.3f} noise '
                              f'{mf["tv_noise"]:.3f} unif {mf["tv_uniform"]:.3f} sp {mf["spearman"]:.2f} '
                              f'p_next {mf["p_next"]:.3f} rho_next {mf["rho_next"]:.3f} beyond '
                              f'{mf["mass_beyond_next"]:.3f} top1=next {int(mf["top1_is_next"])} '
                              f'H_p {mf["H_p"]:.2f} H_rho {mf["H_rho"]:.2f} succ {np.mean(succ):.2f} '
                              f'({time.time() - t0:.0f} s)', flush=True)
                    observation = restore(snap)
                observation, _, terminated, truncated, info = eval_env.step(act(policy, observation))
                done = terminated or truncated
                step += 1
            ep_success[str(task)].append(int(bool(info.get('success', False))))
            print(f'task {task} episode {ep}: success={ep_success[str(task)][-1]} length={step}', flush=True)

    def flat(recs, nn_name):
        return [dict(r['metrics'][nn_name]) for r in recs]

    aggregates = {}
    for nn_name in nn_modes:
        agg = {'all': summarise(flat(probes, nn_name))}
        for setting in GOAL_SETTINGS:
            sel = [r for r in probes if r['goal'] == setting]
            agg[setting] = summarise(flat(sel, nn_name))
            for task in tasks:
                st = [r for r in sel if r['task'] == int(task)]
                agg[f'{setting}/task{task}'] = summarise(flat(st, nn_name))
        aggregates[nn_name] = agg

    report = {
        'tool': 'diag_measure_fullness',
        'env_prefix': prefix, 'tasks': tasks, 'episodes_per_task': n_ep, 'probe_ts': list(probe_ts),
        'R': R, 'T': T, 'n_ref': n_ref, 'n_ref_cols': n_ref_cols, 'seed': base_seed,
        'task_goal_pick': task_goal_pick, 'goal_settings': list(GOAL_SETTINGS), 'nn_modes': list(nn_modes),
        'gamma': gamma, 'measure_temp': temp, 'measure_loss': loss, 'gpi_num_u': K, 'u_clip': u_clip,
        'runs': runs, 'restore_epoch': int(cfg.restore_epoch),
        'train_seed': {s: prov[s].get('train_seed') for s in runs},
        'forced_agent_keys': FORCED_AGENT_KEYS, 't_max': t_max,
        'determinism_max_dev': det_dev,
        'main_episode_success': ep_success,
        'x_indices': x_idx.tolist(), 'obs_std': std.tolist(),
        'probes': probes, 'aggregates': aggregates,
        'agent_config': json.loads(json.dumps(configs['a'].to_dict(), default=str)),
        'config_provenance': prov,
        'wall_seconds': round(time.time() - t_start, 1),
    }
    cols = ('tv', 'tv_noise', 'tv_uniform', 'tv_b', 'spearman', 'top1_is_next', 'p_next', 'rho_next',
            'mass_beyond_next', 'H_p', 'H_rho')
    print(f'\n{"group":22s}' + ''.join(f'{k:>17s}' for k in cols))
    for nn_name in nn_modes:
        for grp in ('task', 'random'):
            s = aggregates[nn_name][grp]
            row = ''.join(f'{s[k]["mean"]:9.3f}±{s[k]["se"]:6.3f} ' if k in s else f'{"-":>17s}' for k in cols)
            print(f'{nn_name + "/" + grp:22s}{row}  (n={s["n"]})')
    write_report(report, cfg, 'diag_measure_fullness.json')


if __name__ == '__main__':
    import hydra

    @hydra.main(version_base=None, config_path='../configs', config_name='config')
    def _main(cfg):
        run(cfg)

    _main()
