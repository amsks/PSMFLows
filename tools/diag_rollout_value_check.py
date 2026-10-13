"""Which critic ranks candidate actions the way real rollouts do? (2026-10-13, task 2b part 2 of
mailbox/tasks/2026-10-09-design-decisions.md)

Two psmgoal checkpoints trained identically except for the bootstrap latent: critic A (default the
flowbc actor run psmgoal_ja_fbc3_pt_cube) and critic B (the run whose actor is pinned to BC,
psmgoal_ja_fbcBC_pt_cube). Both are built exactly as tools/eval_checkpoint.py builds them (own
flags.json plus the typed CLI overrides, preimage splice, np.random.seed(seed), the same relabel
batch for both) and both read the candidates with `_hgoal_each_score`.

Main episodes are acted by the continuation policy (+cont=a: best-of-64 with critic A, the deployed
policy; +cont=b: with critic B). At probe steps t in +probe_ts of each episode:

  1. snapshot the simulator (mj_getState, mjSTATE_INTEGRATION);
  2. draw 64 clipped prior latents and score them with A and B;
  3. candidates: A's top, B's top and random others, n_cand in total (distinct indices);
  4. per candidate, n_rep branches: restore the snapshot, execute the candidate's decoded action,
     then act with the continuation policy until the task succeeds or the episode's time limit
     (the env's max_episode_steps minus t). Outcome: success (0/1) and gamma^k with k the number of
     steps to success (0 on failure), gamma = the run's goal_discount;
  5. restore the snapshot and continue the main episode.

Per probe state over the candidates: Spearman of each critic's score with the mean discounted
outcome; the outcome of A's top and of B's top; the best candidate outcome. The split-half Spearman
between the first and second half of the branches is the noise ceiling for the critics'
correlations. A determinism check replays one branch's action sequence from the snapshot and must
reproduce its observations exactly.

Run one (seed, task) cell (GPU):
  .venv/bin/python tools/diag_rollout_value_check.py agent=psmgoal \
    env_name=cube-single-play-singletask-task1-v0 \
    agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \
    agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz \
    agent.coef_source=hgoal_each agent.acting=gpi \
    restore_path=<critic A run> restore_epoch=500000 +restore_path_b=<critic B run> \
    eval_episodes=8 seed=0 report_out=<cell json>

Pool cells:
  .venv/bin/python tools/diag_rollout_value_check.py aggregate <out.json> <cell1.json> ...
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax

PROBE_TS = (0, 15, 30, 45, 60, 90)


def spearman(a, b):
    """Spearman correlation of two 1-d arrays, average ranks on ties; nan if either is constant."""
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float('nan')


def pick_candidates(qa, qb, n_cand, rng):
    """Indices: A's argmax, B's argmax (if different), then random distinct others."""
    first = [int(np.argmax(qa))]
    if int(np.argmax(qb)) != first[0]:
        first.append(int(np.argmax(qb)))
    rest = [i for i in rng.permutation(qa.shape[0]) if i not in first]
    return np.asarray(first + rest[:n_cand - len(first)], np.int64)


def probe_metrics(qa, qb, disc, succ):
    """qa, qb (C,) critic scores; disc, succ (C, R) branch outcomes. Candidate 0 = A's top."""
    m_disc = disc.mean(1)
    ia, ib = int(np.argmax(qa)), int(np.argmax(qb))
    half = disc.shape[1] // 2
    out = {
        'spearman_a': spearman(qa, m_disc),
        'spearman_b': spearman(qb, m_disc),
        'split_half': spearman(disc[:, :half].mean(1), disc[:, half:].mean(1)) if half else float('nan'),
        'a_top_disc': float(m_disc[ia]), 'b_top_disc': float(m_disc[ib]),
        'a_top_succ': float(succ[ia].mean()), 'b_top_succ': float(succ[ib].mean()),
        'best_disc': float(m_disc.max()), 'mean_disc': float(m_disc.mean()),
        'mean_succ': float(succ.mean()), 'tops_differ': float(ia != ib),
        'outcome_spread': float(m_disc.std()),
    }
    return out


def summarise(probes):
    if not probes:
        return {'n_probes': 0}
    out = {'n_probes': len(probes)}
    for k in probes[0]:
        v = np.asarray([p[k] for p in probes], np.float64)
        out[k] = float(np.nanmean(v))
        out[k + '_n'] = int(np.isfinite(v).sum())
    return out


def aggregate(paths, out_path):
    """Mean over tasks per seed, then mean and SE over seeds."""
    cells = []
    for p in paths:
        with open(p) as f:
            cells.append(json.load(f))
    by_seed = {}
    for c in cells:
        by_seed.setdefault(c['seed_tag'], []).append(c['summary'])
    keys = ('spearman_a', 'spearman_b', 'split_half', 'a_top_disc', 'b_top_disc', 'a_top_succ',
            'b_top_succ', 'best_disc', 'mean_disc', 'mean_succ', 'tops_differ', 'outcome_spread')
    per_seed = {s: {k: float(np.nanmean([c[k] for c in cs])) for k in keys} for s, cs in by_seed.items()}
    table = {}
    for k in keys:
        v = np.asarray([per_seed[s][k] for s in per_seed])
        table[k] = {'mean': float(v.mean()),
                    'se': float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None}
    report = {'n_cells': len(cells), 'n_seeds': len(per_seed), 'per_seed': per_seed, 'table': table}
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    for k, v in table.items():
        print(f'{k:16s} {v["mean"]:.3f}' + (f' ± {v["se"]:.3f}' if v['se'] is not None else ''))


def load_pair(cfg):
    """Critics A (cfg.restore_path) and B (cfg.restore_path_b), built as tools/eval_checkpoint.py
    builds them, with their eval goals inferred on the same relabel batch. Returns (ag, eval_env)
    with ag = {'a': agent, 'b': agent}. Also used by diag_failure_stages.py and diag_segment_swap.py."""
    import ml_collections
    from omegaconf import OmegaConf

    from agents import agents
    from envs.env_utils import make_env_and_datasets
    from main import _lists_to_tuples
    from tools.eval_checkpoint import _cli_agent_keys, merge_run_config
    from utils.datasets import Dataset
    from utils.flax_utils import restore_agent
    from utils.flow_inversion import load_augmented_dataset, repair_invalid_preimages

    base_seed = int(cfg.seed)
    run_b = cfg.get('restore_path_b', None)
    assert run_b, 'pass +restore_path_b=<critic B run>'

    cli_agent = OmegaConf.to_container(cfg.agent, resolve=True)
    configs = {}
    for side, rp in (('a', str(cfg.restore_path)), ('b', str(run_b))):
        merged, _ = merge_run_config(cli_agent, rp, _cli_agent_keys())
        configs[side] = ml_collections.ConfigDict(_lists_to_tuples(merged))
        assert configs[side]['agent_name'] == 'psmgoal'
        assert configs[side]['acting'] == 'gpi' and configs[side]['coef_source'] == 'hgoal_each', \
            'both critics are read with acting=gpi coef_source=hgoal_each'

    np.random.seed(base_seed)
    _, eval_env, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    train_dataset = dict(train_dataset)
    aug, _ = repair_invalid_preimages(load_augmented_dataset(configs['a']['preimage_path']))
    for k in aug:
        if k.startswith('noise_preimage'):
            train_dataset[k] = aug[k]
    ds = Dataset.create(**train_dataset)
    ds.return_preimage_noise = True
    ds.preimage_point_mode = bool(configs['a'].get('use_point_preimage', False))
    ex = ds.sample(1)
    zb = ds.sample(min(ds.size, int(cfg.get('eval_relabel_size', 10000))))
    rew = zb['rewards'] + float(cfg.get('eval_reward_shift', 1.0))
    ag = {}
    for side, rp in (('a', str(cfg.restore_path)), ('b', str(run_b))):
        a = agents['psmgoal'].create(base_seed, ex['observations'], ex['actions'], configs[side])
        a = restore_agent(a, rp, cfg.restore_epoch)
        ag[side] = a.infer_eval_goals(zb, rew)
    return ag, eval_env


class SimState:
    """Snapshot and restore of the MuJoCo integration state of an OGBench manipulation env."""

    def __init__(self, eval_env):
        import mujoco
        self.mujoco = mujoco
        self.env = eval_env.unwrapped
        self.model, self.data = self.env._model, self.env._data
        self.spec = mujoco.mjtState.mjSTATE_INTEGRATION

    def snapshot(self):
        st = np.empty(self.mujoco.mj_stateSize(self.model, self.spec))
        self.mujoco.mj_getState(self.model, self.data, st, self.spec)
        return st, bool(self.env._success)

    def restore(self, snap):
        self.mujoco.mj_setState(self.model, self.data, snap[0], self.spec)
        self.mujoco.mj_forward(self.model, self.data)
        self.env._success = snap[1]
        return self.env.compute_observation()


def run(cfg):
    import jax
    import jax.numpy as jnp

    base_seed = int(cfg.seed)
    n_ep = int(cfg.eval_episodes)
    n_cand = int(cfg.get('n_cand', 8))
    n_rep = int(cfg.get('n_rep', 2))
    probe_ts = tuple(int(t) for t in cfg.get('probe_ts', PROBE_TS))
    cont = str(cfg.get('cont', 'a'))
    assert cont in ('a', 'b')
    run_b = cfg.get('restore_path_b', None)
    ag, eval_env = load_pair(cfg)

    c = ag['a'].config
    K, d_a, u_clip = int(c['gpi_num_u']), int(c['action_dim']), float(c['u_clip'])
    gamma = float(c['goal_discount'])

    @jax.jit
    def scores(agent, obs, u):
        o = jnp.broadcast_to(obs, (u.shape[0], obs.shape[-1]))
        return agent._hgoal_each_score(o, u, agent.eval_goals, agent.eval_goal_w, agent.eval_ref)

    @jax.jit
    def decode1(agent, obs, u):
        return agent.decode(obs[None], u[None])[0]

    policy = ag[cont]
    keys = {'k': jax.random.PRNGKey(base_seed)}

    def next_key():
        keys['k'], sub = jax.random.split(keys['k'])
        return sub

    def act(obs):
        return np.clip(np.asarray(policy.sample_actions(observations=obs, seed=next_key(), temperature=0)), -1, 1)

    sim = SimState(eval_env)
    env, snapshot, restore = sim.env, sim.snapshot, sim.restore
    t_max = int(eval_env.spec.max_episode_steps) if eval_env.spec and eval_env.spec.max_episode_steps else 200

    def branch(snap, first_action, horizon, record=False):
        """Restore, execute first_action, then act with the policy. Returns (success, k, obs list, actions)."""
        restore(snap)
        a, obs_seq, act_seq = first_action, [], []
        for k in range(1, horizon + 1):
            ob, _, _, _, info = env.step(a)
            if record:
                obs_seq.append(np.asarray(ob))
                act_seq.append(np.asarray(a))
            if bool(info['success']):
                return True, k, obs_seq, act_seq
            a = act(ob)
        return False, horizon, obs_seq, act_seq

    rng = np.random.default_rng(base_seed + 7919)
    probes, ep_success = [], []
    det_dev = None
    t0 = time.time()
    eval_env.reset(seed=base_seed)
    for ep in range(n_ep):
        observation, info = eval_env.reset()
        done, step = False, 0
        while not done:
            if step in probe_ts and step < t_max - 1:
                snap = snapshot()
                u = jnp.clip(jax.random.normal(next_key(), (K, d_a)), -u_clip, u_clip)
                obs_j = jnp.asarray(observation)
                qa = np.asarray(scores(ag['a'], obs_j, u))
                qb = np.asarray(scores(ag['b'], obs_j, u))
                idx = pick_candidates(qa, qb, n_cand, rng)
                horizon = t_max - step
                disc = np.zeros((len(idx), n_rep))
                succ = np.zeros((len(idx), n_rep))
                for ci, j in enumerate(idx):
                    a0 = np.clip(np.asarray(decode1(policy, obs_j, u[j])), -1, 1)
                    for r in range(n_rep):
                        rec = det_dev is None and ci == 0 and r == 0
                        ok, k, obs_seq, act_seq = branch(snap, a0, horizon, record=rec)
                        succ[ci, r] = float(ok)
                        disc[ci, r] = gamma ** k if ok else 0.0
                        if rec:
                            # replay the same action sequence from the snapshot: must match exactly
                            restore(snap)
                            dev = 0.0
                            for a_r, o_r in zip(act_seq, obs_seq):
                                ob2 = env.step(a_r)[0]
                                dev = max(dev, float(np.max(np.abs(np.asarray(ob2) - o_r))))
                            det_dev = dev
                            assert det_dev == 0.0, f'simulator restore is not deterministic: {det_dev}'
                m = probe_metrics(qa[idx], qb[idx], disc, succ)
                m.update({'episode': ep, 't': step})
                probes.append(m)
                observation = restore(snap)
                print(f'ep {ep} t {step}: spearman a {m["spearman_a"]:.2f} b {m["spearman_b"]:.2f} '
                      f'split {m["split_half"]:.2f} top disc a {m["a_top_disc"]:.3f} b {m["b_top_disc"]:.3f} '
                      f'mean succ {m["mean_succ"]:.2f} ({time.time() - t0:.0f} s)', flush=True)
            observation, _, terminated, truncated, info = eval_env.step(act(observation))
            done = terminated or truncated
            step += 1
        ep_success.append(bool(info.get('success', False)))
        print(f'episode {ep}: success={ep_success[-1]} length={step}', flush=True)

    numeric = [{k: v for k, v in p.items() if k not in ('episode', 't')} for p in probes]
    report = {
        'env': str(cfg.env_name), 'run_a': str(cfg.restore_path), 'run_b': str(run_b),
        'restore_epoch': int(cfg.restore_epoch), 'seed': base_seed, 'cont': cont,
        'seed_tag': os.path.basename(str(cfg.restore_path))[:5],
        'n_cand': n_cand, 'n_rep': n_rep, 'probe_ts': list(probe_ts), 'gamma': gamma, 't_max': t_max,
        'determinism_max_dev': det_dev, 'main_episode_success': [int(x) for x in ep_success],
        'probes': probes, 'summary': summarise(numeric), 'wall_seconds': round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(cfg.report_out)), exist_ok=True)
    with open(cfg.report_out, 'w') as f:
        json.dump(report, f, indent=2)
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
