"""CPU tests for the pure half of tools/diag_measure_fullness.py: nearest-neighbour matching,
the discounted occupancy over a reference set, the measure-to-distribution read and the
per-probe comparison metrics, on hand-built vectors with known answers."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.diag_measure_fullness import (
    compare,
    entropy,
    measure_to_dist,
    nearest,
    obs_std,
    occupancy,
    parse_int_list,
    split_half_tv,
    summarise,
    tv,
)


def test_nearest_standardized_and_column_subset():
    X = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 1.0]])
    std = np.array([10.0, 1.0])           # column 0 is 10x wider: (10, 0) is one std unit from (0, 0)
    X_std = X / std
    # (6, 0.9): standardized (0.6, 0.9) -> nearest is (0, 1) at distance sqrt(0.36 + 0.01)
    assert nearest(X_std, np.array([6.0, 0.9]), std).tolist() == [2]
    # on column 0 only the same query is nearest to (10, 0) (0.6 vs 0.4)
    assert nearest(X_std, np.array([6.0, 0.9]), std, slice(0, 1)).tolist() == [1]
    assert nearest(X_std, np.array([[0.1, 0.0], [9.0, 0.0]]), std).tolist() == [0, 1]


def test_obs_std_constant_dimension_is_one():
    s = obs_std(np.array([[1.0, 0.0], [3.0, 0.0]]))
    np.testing.assert_allclose(s, [1.0, 1.0])


def test_occupancy_known_trajectory():
    gamma = 0.5
    # one rollout visiting x=2, x=0, x=0: weights 1, .5, .25 over 1.75
    rho = occupancy([np.array([2, 0, 0])], n_ref=3, gamma=gamma)
    np.testing.assert_allclose(rho, [0.75 / 1.75, 0.0, 1.0 / 1.75])
    assert abs(rho.sum() - 1.0) < 1e-12
    # a second rollout of one step at x=1 is averaged in with weight 1/2
    rho2 = occupancy([np.array([2, 0, 0]), np.array([1])], n_ref=3, gamma=gamma)
    np.testing.assert_allclose(rho2, [0.5 * 0.75 / 1.75, 0.5, 0.5 / 1.75])
    # an empty rollout contributes nothing but still counts in the average
    rho3 = occupancy([np.array([1]), np.array([], np.int64)], n_ref=2, gamma=gamma)
    np.testing.assert_allclose(rho3, [0.0, 0.5])


def test_measure_to_dist():
    m = np.array([0.0, np.log(3.0)])
    np.testing.assert_allclose(measure_to_dist(m, 'softmax', 1.0), [0.25, 0.75])
    np.testing.assert_allclose(measure_to_dist(np.array([0.0, 2 * np.log(3.0)]), 'softmax', 2.0), [0.25, 0.75])
    # measure_loss=pool is a softmax over candidate states: read like softmax, never the clip rule
    np.testing.assert_allclose(measure_to_dist(m, 'pool', 1.0), [0.25, 0.75])
    np.testing.assert_allclose(measure_to_dist(np.array([-1.0, -2.0]), 'pool', 1.0),
                               measure_to_dist(np.array([-1.0, -2.0]), 'softmax', 1.0))
    np.testing.assert_allclose(measure_to_dist(np.array([-1.0, 1.0, 3.0]), 'squared', 1.0), [0.0, 0.25, 0.75])
    np.testing.assert_allclose(measure_to_dist(np.array([-1.0, -2.0]), 'squared', 1.0), [0.5, 0.5])


def test_tv_entropy_and_compare():
    p = np.array([0.7, 0.2, 0.1, 0.0])
    rho = np.array([0.1, 0.5, 0.0, 0.4])
    assert abs(tv(p, rho) - 0.5 * (0.6 + 0.3 + 0.1 + 0.4)) < 1e-12
    assert abs(entropy(np.array([0.5, 0.5, 0.0])) - np.log(2.0)) < 1e-12
    m = compare(p, rho, next_idx=0)
    assert m['p_top1_idx'] == 0 and m['top1_is_next'] == 1.0
    assert abs(m['p_top1'] - 0.7) < 1e-12 and abs(m['p_next'] - 0.7) < 1e-12 and abs(m['rho_next'] - 0.1) < 1e-12
    # shared mass beyond the next state: min(0.2, 0.5) + min(0.1, 0) + min(0, 0.4) = 0.2
    assert abs(m['mass_beyond_next'] - 0.2) < 1e-12
    assert m['rho_support'] == 3
    assert abs(m['tv_uniform'] - tv(np.full(4, 0.25), rho)) < 1e-12
    assert 'tv_b' not in m
    # a rollout that lands where p points: next_idx = 1, p's top is 0 -> not the detector signature
    m1 = compare(p, rho, next_idx=1)
    assert m1['top1_is_next'] == 0.0 and abs(m1['mass_beyond_next'] - (0.1 + 0.0 + 0.0)) < 1e-12
    # critic B at the same probe
    mb = compare(p, rho, next_idx=0, p_b=rho)
    assert abs(mb['tv_b']) < 1e-12 and abs(mb['tv_a_b'] - tv(p, rho)) < 1e-12
    assert abs(mb['p_b_next'] - 0.1) < 1e-12 and mb['top1_b_is_next'] == 0.0
    # perfectly ranked distributions: spearman 1 over all and over the support
    q = np.array([0.4, 0.3, 0.2, 0.1])
    mq = compare(q, q, next_idx=0)
    assert abs(mq['spearman'] - 1.0) < 1e-12 and abs(mq['spearman_support'] - 1.0) < 1e-12


def test_split_half_tv():
    gamma = 0.9
    same = [np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1])]
    assert abs(split_half_tv(same, 2, gamma)) < 1e-12
    diff = [np.array([0]), np.array([0]), np.array([1]), np.array([1])]
    assert abs(split_half_tv(diff, 2, gamma) - 1.0) < 1e-12
    assert np.isnan(split_half_tv([np.array([0])], 2, gamma))
    # an odd rollout is dropped: halves are [0], [1]
    assert abs(split_half_tv([np.array([0]), np.array([1]), np.array([0])], 2, gamma) - 1.0) < 1e-12


def test_summarise_and_parse():
    recs = [{'tv': 0.2, 'H_p': float('nan')}, {'tv': 0.4, 'H_p': 1.0}]
    s = summarise(recs, keys=('tv', 'H_p', 'tv_b'))
    assert s['n'] == 2 and abs(s['tv']['mean'] - 0.3) < 1e-12 and s['tv']['n'] == 2
    assert abs(s['tv']['se'] - np.std([0.2, 0.4], ddof=1) / np.sqrt(2)) < 1e-12
    assert s['H_p']['n'] == 1 and s['H_p']['se'] == 0.0 and 'tv_b' not in s
    assert parse_int_list(None, (0, 20)) == [0, 20]
    assert parse_int_list('0,40', (0,)) == [0, 40]
    assert parse_int_list('[1 2 3]', (0,)) == [1, 2, 3]
    assert parse_int_list([4, 5], (0,)) == [4, 5]


# ------------------------------------------------------------------ run(): ref_set and the bank index
# The tool's run() on a fake 6-dim environment (deterministic dynamics, a snapshot that copies the
# observation) and a saved CPU agent (allow_untrained_flow), so the pool / bank paths and the report
# schema are pinned without a GPU, a dataset or MuJoCo. Run in its own process:
#   JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_diag_measure_fullness.py -q

import json
import types

import pytest

# top-level report keys of a goal-index run before 2026-10-19 (ref_set / bank); pinned by test (d)
REPORT_KEYS_BEFORE = frozenset((
    'tool', 'env_prefix', 'tasks', 'episodes_per_task', 'probe_ts', 'R', 'T', 'n_ref', 'n_ref_cols', 'seed',
    'task_goal_pick', 'rollout_policy', 'probe_policy', 'goal_settings', 'nn_modes', 'gamma', 'measure_temp',
    'measure_loss', 'gpi_num_u', 'u_clip', 'runs', 'restore_epoch', 'train_seed', 'forced_agent_keys', 't_max',
    'determinism_max_dev', 'main_episode_success', 'x_indices', 'obs_std', 'probes', 'aggregates',
    'agent_config', 'config_provenance', 'wall_seconds'))
PROBE_KEYS_BEFORE = frozenset((
    'task', 'episode', 't', 'goal', 'goal_source_index', 'goal_share_top', 'goal_share_mean', 'u_norm',
    'rollout_success', 'rollout_length', 'rollout_terminated', 'next_state_dev_across_rollouts', 'M_mean',
    'M_max', 'M_min', 'metrics'))
N_ROWS, POOL_P, EPOCH = 64, 16, 5


def _fake_dataset(n=N_ROWS, seed=3):
    from tests.test_psmgoal_joint_actor import DA, OB
    rng = np.random.default_rng(seed)
    rewards = -np.ones((n,), np.float32)
    rewards[:8] = 0.0                     # +1 shift at eval: 8 rewarding rows, 56 non-rewarding
    terminals = np.zeros((n,), np.float32)
    terminals[-1] = 1.0
    return {
        'observations': rng.standard_normal((n, OB)).astype(np.float32),
        'next_observations': rng.standard_normal((n, OB)).astype(np.float32),
        'actions': np.clip(rng.standard_normal((n, DA)), -1, 1).astype(np.float32),
        'noise_preimage': rng.standard_normal((n, DA)).astype(np.float32),
        'rewards': rewards, 'terminals': terminals, 'masks': 1.0 - terminals,
    }


class _FakeEnv:
    """Deterministic 6-dim dynamics: ob += 0.1 * [a, a]; truncates after max_steps; never succeeds."""

    def __init__(self, ob_dim, max_steps=3):
        self.ob_dim, self.max_steps = ob_dim, max_steps
        self.spec = types.SimpleNamespace(max_episode_steps=max_steps)
        self.unwrapped = self
        self.ob, self.t = np.zeros(ob_dim, np.float32), 0

    def reset(self, seed=None, options=None):
        self.ob, self.t = np.full(self.ob_dim, 0.5, np.float32), 0
        return self.ob.copy(), {}

    def step(self, a):
        a = np.asarray(a, np.float32).reshape(-1)
        self.ob = (self.ob + 0.1 * np.concatenate([a, a])[:self.ob_dim]).astype(np.float32)
        self.t += 1
        return self.ob.copy(), 0.0, False, self.t >= self.max_steps, {'success': False}


class _FakeSim:
    def __init__(self, eval_env):
        self.env = eval_env.unwrapped

    def snapshot(self):
        return self.env.ob.copy(), self.env.t

    def restore(self, snap):
        self.env.ob, self.env.t = snap[0].copy(), snap[1]
        return self.env.ob.copy()


def _pool_rows(data):
    from utils.psm_common import draw_pool_indices
    rows = draw_pool_indices(np.arange(data['next_observations'].shape[0]), POOL_P, 0)
    return data['next_observations'][rows], rows


def _checkpoint(tmp_path, name, agent):
    """A run dir as main.py leaves it: params_<EPOCH>.pkl and flags.json with the agent block."""
    import flax

    from utils.flax_utils import save_agent
    d = tmp_path / name
    d.mkdir()
    save_agent(agent, str(d), EPOCH)
    flags = {'seed': 0, 'agent': json.loads(json.dumps(flax.core.unfreeze(agent.config), default=str))}
    (d / 'flags.json').write_text(json.dumps(flags))
    return str(d)


def _run(monkeypatch, tmp_path, run_dir, data, run_dir_b=None, **over):
    from omegaconf import OmegaConf

    import tools.diag_measure_fullness as mod
    from envs import env_utils
    from tests.test_psmgoal_joint_actor import OB, _cfg
    env = _FakeEnv(OB)
    monkeypatch.setattr(env_utils, 'make_env_and_datasets', lambda name, frame_stack=None, **kw: (None, env, data, None))
    monkeypatch.setattr(mod, 'SimState', _FakeSim)
    out = tmp_path / f'report_{len(list(tmp_path.iterdir()))}.json'
    cfg = {'seed': 0, 'env_name': 'fake-play', 'restore_path': run_dir, 'restore_epoch': EPOCH, 'frame_stack': None,
           'eval_relabel_size': N_ROWS, 'eval_reward_shift': 1.0, 'report_out': str(out),
           'agent': _cfg().to_dict(), 'tasks': '1', 'episodes': 1, 'probe_ts': [0], 'R': 1, 'T': 2, 'n_ref': 8}
    if run_dir_b:
        cfg['restore_path_b'] = run_dir_b
    cfg.update(over)
    mod.run(OmegaConf.create(cfg))
    with open(out) as f:
        return json.load(f)


def _goal_pool_agent(data):
    from tests.test_psmgoal_pool_loss import _goal_over, _pool_agent
    pool_obs, rows = _pool_rows(data)
    return _pool_agent(pool_obs, indices=rows, **_goal_over())


def _bank_pool_agent(data):
    from tests.test_psmgoal_pool_loss import _pool_agent
    pool_obs, rows = _pool_rows(data)
    return _pool_agent(pool_obs, indices=rows, policy_index='bank', train_actor=False, pool_eval_size=16)


def _goal_softmax_agent():
    from tests.test_psmgoal_bootstrap_delta_scale import OVER
    from tests.test_psmgoal_joint_actor import _agent
    return _agent(**OVER)


def test_ref_set_pool_uses_the_agents_pool_states_as_x(monkeypatch, tmp_path):
    data = _fake_dataset()
    agent = _goal_pool_agent(data)
    rep = _run(monkeypatch, tmp_path, _checkpoint(tmp_path, 'goal_pool', agent), data, ref_set='pool')
    assert rep['ref_set'] == 'pool' and rep['n_ref'] == POOL_P and rep['measure_loss'] == 'pool'
    assert rep['policy_index'] == 'goal' and rep['goal_settings'] == ['task', 'random']
    x = data['next_observations'][np.asarray(rep['x_indices'])]
    assert np.array_equal(x, np.asarray(agent.pool_states, np.float32))           # X is exactly the pool
    assert np.array_equal(np.asarray(rep['x_indices']), np.asarray(agent.pool_indices))
    for r in rep['probes']:
        for nn in rep['nn_modes']:
            assert 0 <= r['metrics'][nn]['next_idx'] < POOL_P and 0 <= r['metrics'][nn]['p_top1_idx'] < POOL_P
    assert rep['determinism_max_dev'] == 0.0 and len(rep['probes']) == 2


def test_ref_set_pool_refuses_a_non_pool_run(monkeypatch, tmp_path):
    data = _fake_dataset()
    run_dir = _checkpoint(tmp_path, 'goal_softmax', _goal_softmax_agent())
    with pytest.raises(AssertionError, match='ref_set=pool'):
        _run(monkeypatch, tmp_path, run_dir, data, ref_set='pool')


@pytest.mark.parametrize('over', [
    {'ref_set': 'random', 'rollout_policy': 'bc'},
    {'ref_set': 'pool', 'rollout_policy': 'bon64', 'probe_policy': 'bon64'},
])
def test_bank_index_reads_p_under_bank_w_and_probes_with_pool_reward(monkeypatch, tmp_path, over):
    import jax

    from agents.psmgoal import PSMGoalAgent
    data = _fake_dataset()
    agent = _bank_pool_agent(data)
    calls = {'pool_reward': 0, 'hgoal_each': 0}
    orig_pool, orig_hgoal = PSMGoalAgent._pool_reward_score, PSMGoalAgent._hgoal_each_score

    def spy_pool(self, *a, **k):
        calls['pool_reward'] += 1
        return orig_pool(self, *a, **k)

    def spy_hgoal(self, *a, **k):
        calls['hgoal_each'] += 1
        return orig_hgoal(self, *a, **k)

    monkeypatch.setattr(PSMGoalAgent, '_pool_reward_score', spy_pool)
    monkeypatch.setattr(PSMGoalAgent, '_hgoal_each_score', spy_hgoal)
    jax.clear_caches()                     # the spies must see the traces, not a cached jit
    rep = _run(monkeypatch, tmp_path, _checkpoint(tmp_path, 'bank_pool', agent), data, **over)
    assert rep['policy_index'] == 'bank' and rep['goal_settings'] == ['bank']
    assert rep['forced_agent_keys'] == {'coef_source': 'pool_reward', 'acting': 'gpi'}
    assert rep['agent_config']['coef_source'] == 'pool_reward' and rep['agent_config']['acting'] == 'gpi'
    assert rep['ref_set'] == over['ref_set'] and rep['n_ref'] == (POOL_P if over['ref_set'] == 'pool' else 8)
    # the coefficient of every read is the bank's single w
    np.testing.assert_allclose(np.asarray(rep['bank_w']['a']), np.asarray(agent.bank_w())[0], rtol=1e-5)
    assert calls['pool_reward'] > 0 and calls['hgoal_each'] == 0
    assert len(rep['probes']) == 1
    r = rep['probes'][0]
    assert r['goal'] == 'bank' and r['goal_source_index'] is None
    assert r['goal_share_top'] is None and r['goal_share_mean'] is None
    assert set(r) == PROBE_KEYS_BEFORE
    assert r['rollout_length'] == [2] and np.isfinite(r['metrics']['full']['tv'])
    assert set(rep['aggregates']['full']) == {'all', 'bank', 'bank/task1'}
    assert rep['aggregates']['full']['bank']['n'] == 1
    assert rep['determinism_max_dev'] == 0.0


def test_bank_index_refuses_a_goal_index_critic_b(monkeypatch, tmp_path):
    data = _fake_dataset()
    a = _checkpoint(tmp_path, 'bank_a', _bank_pool_agent(data))
    b = _checkpoint(tmp_path, 'goal_b', _goal_pool_agent(data))
    with pytest.raises(AssertionError, match='policy_index=goal'):
        _run(monkeypatch, tmp_path, a, data, run_dir_b=b)


def test_goal_index_report_keys_are_unchanged(monkeypatch, tmp_path):
    data = _fake_dataset()
    rep = _run(monkeypatch, tmp_path, _checkpoint(tmp_path, 'goal_softmax', _goal_softmax_agent()), data)
    assert set(rep) - REPORT_KEYS_BEFORE == {'ref_set', 'policy_index'}
    assert REPORT_KEYS_BEFORE <= set(rep) and 'bank_w' not in rep
    assert rep['ref_set'] == 'random' and rep['policy_index'] == 'goal' and rep['n_ref'] == 8
    assert rep['goal_settings'] == ['task', 'random']
    assert rep['forced_agent_keys'] == {'coef_source': 'hgoal_each', 'acting': 'gpi'}
    assert [r['goal'] for r in rep['probes']] == ['task', 'random']
    for r in rep['probes']:
        assert set(r) == PROBE_KEYS_BEFORE
        assert isinstance(r['goal_source_index'], int) and isinstance(r['goal_share_top'], float)
    assert set(rep['aggregates']['full']) == {'all', 'task', 'random', 'task/task1', 'random/task1'}
    assert set(rep['aggregates']) == {'full', 'cube_xyz'}
