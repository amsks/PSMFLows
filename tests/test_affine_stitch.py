"""Guard stitch boundaries, reward exclusion, inference isolation and run provenance."""

import importlib
import json

import numpy as np
import pytest


def data_api():
    return importlib.import_module('tools.affine_stitch.data')


def fixture_npz(path, terminals=None):
    np.savez(path, observations=np.array([[0], [1], [2], [10], [11], [12]], np.float32),
             actions=np.zeros((6, 1), np.float32),
             terminals=np.array([0, 0, 1, 0, 0, 1] if terminals is None else terminals, bool),
             rewards=np.full(6, 999.0))


def test_native_pairing_excludes_terminal_sentinels_and_rewards(tmp_path):
    path = tmp_path / 'train.npz'
    fixture_npz(path)
    data, record = data_api().load_file(path, data_api().sha256(path), expected_transitions=4, episode_rows=3)
    np.testing.assert_array_equal(data['observations'][:, 0], [0, 1, 10, 11])
    np.testing.assert_array_equal(data['next_observations'][:, 0], [1, 2, 11, 12])
    batch = data_api().Replay(data, 2).sample(10)
    assert set(batch) == {'observations', 'actions', 'next_observations', 'index'}
    np.testing.assert_array_equal(batch['observations'], data['observations'][batch['index']])
    assert record['transitions'] == 4


def test_changed_checksum_or_terminal_layout_is_rejected(tmp_path):
    path = tmp_path / 'train.npz'
    fixture_npz(path)
    with pytest.raises(ValueError, match='checksum'):
        data_api().load_file(path, 'wrong', expected_transitions=4, episode_rows=3)
    fixture_npz(path, [0, 1, 0, 0, 0, 1])
    with pytest.raises(ValueError, match='terminal'):
        data_api().load_file(path, data_api().sha256(path), expected_transitions=4, episode_rows=3)


def test_inference_sampling_does_not_advance_training_rng():
    arrays = {k: np.arange(100)[:, None] for k in ['observations', 'actions', 'next_observations']}
    train, control = data_api().Replay(arrays, 7), data_api().Replay(arrays, 7)
    train.sample(5)
    control.sample(5)
    inference = train.fork(0)
    inference.sample(4096)
    np.testing.assert_array_equal(train.sample(20)['index'], control.sample(20)['index'])
    np.testing.assert_array_equal(train.fork(0).sample(30)['index'], control.fork(0).sample(30)['index'])


def test_bounded_proto_correction_preserves_draws_and_rejects_wrong_range():
    from tools.affine_stitch.run import correct_proto_table

    released = np.array([[-1.9, -0.2], [-0.7, -1.1]], np.float32)
    np.testing.assert_array_equal(correct_proto_table(released, 'bounded'), released + 1)
    np.testing.assert_array_equal(correct_proto_table(released, 'released'), released)
    with pytest.raises(ValueError, match='range'):
        correct_proto_table(np.array([[1.5]], np.float32), 'bounded')


@pytest.mark.parametrize('changed', ['source_manifest', 'dataset_manifest'])
def test_sidecar_changes_cannot_rebind_checkpoint(tmp_path, changed):
    flags = {}
    for name in ['source_manifest', 'dataset_manifest']:
        path = tmp_path / f'{name}.json'
        path.write_text('{}')
        flags[f'{name}_sha256'] = data_api().sha256(path)
    data_api().verify_run_sidecars(tmp_path, flags)
    (tmp_path / f'{changed}.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match=changed):
        data_api().verify_run_sidecars(tmp_path, flags)


def test_snapshot_code_change_rejected(tmp_path):
    from tools.affine_stitch.run import verify_snapshot

    path = tmp_path / 'affine_agent.py'
    path.write_text('x = 1\n')
    (tmp_path / 'source_manifest.json').write_text(json.dumps({'files': {path.name: data_api().sha256(path)}}))
    verify_snapshot(tmp_path)
    path.write_text('x = 2\n')
    with pytest.raises(ValueError, match='hash'):
        verify_snapshot(tmp_path)


def test_checkpoint_restores_optimizer_rng_and_rejects_changed_flags(tmp_path):
    import flax.struct
    import jax.numpy as jnp

    from tools.affine_stitch.run import restore_checkpoint, save_checkpoint

    @flax.struct.dataclass
    class State:
        params: object
        opt_state: object

    state = State(jnp.array([1.0]), {'moment': jnp.array([0.25])})
    rng = np.random.default_rng(13)
    path = tmp_path / 'checkpoint.pkl'
    save_checkpoint(path, state, 42, rng, 'abc')
    restored, step, restored_rng = restore_checkpoint(path, state, 'abc')
    assert step == 42
    np.testing.assert_array_equal(restored.opt_state['moment'], [0.25])
    np.testing.assert_array_equal(rng.integers(100, size=10), restored_rng.integers(100, size=10))
    with pytest.raises(ValueError, match='flags'):
        restore_checkpoint(path, state, 'wrong')


def test_seed_statistics_averages_tasks_before_interval():
    from tools.affine_stitch.report import seed_statistics

    result = seed_statistics([[0, 1, 0, 1, 0.5], [1, 0, 1, 0, 0.5], [0.5] * 5])
    assert result['seed_task_means'] == [0.5, 0.5, 0.5]
    assert result['ci95_halfwidth'] == 0


def test_environment_reset_is_reproducible_without_changing_global_rng():
    from tools.affine_stitch.run import make_env, seeded_reset

    env = make_env(1)
    np.random.seed(77)
    expected = np.random.RandomState(77).rand(8)
    try:
        first, info1 = seeded_reset(env, 13)
        np.testing.assert_array_equal(np.random.rand(8), expected)
        second, info2 = seeded_reset(env, 13)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(info1['goal'], info2['goal'])
        assert info1['goal'].shape == (29,)
    finally:
        env.close()


@pytest.mark.parametrize('width', [8, 32])
def test_snapshot_real_agent_inference_and_degenerate_feature_rejection(tmp_path, width):
    import os
    import subprocess
    import sys

    from tools.affine_stitch.materialize import materialize

    code = tmp_path / 'code'
    manifest = materialize(code)
    assert 'affine_agent.py' in manifest['files']
    script = '''
import json
import os
import sys
from tools.affine_stitch.run import reference_config, infer_task, correct_proto_table
from tools.affine_stitch.data import Replay
from affine_agent import AffinePSMAgent
import numpy as np
cfg = reference_config()
cfg.update(batch_size=8, d_dim=4, z_dim=4, max_log_seed=3)
width = int(os.environ['AFFINE_TEST_WIDTH'])
cfg['measure'].update(hidden_dim=width, hidden_layers=1, k_dim=2)
cfg['actor'].update(hidden_dim=width, hidden_layers=1, embedding_layers=2)
cfg['inference'].update(num_inference_steps=3, num_actor_inference_steps=2,
                        lagrange_hidden_dim=8, lagrange_hidden_layers=1)
rng = np.random.default_rng(17)
arrays = dict(observations=rng.normal(size=(32, 29)).astype('float32'),
              next_observations=rng.normal(size=(32,29)).astype('float32'),
              actions=rng.uniform(-1,1,size=(32,8)).astype('float32'))
data = Replay(arrays, 0)
agent = AffinePSMAgent.create(0, arrays['observations'][:1], arrays['actions'][:1], cfg)
table, powers = agent.proto
agent = agent.replace(proto=(correct_proto_table(table, 'bounded'), powers))
agent, losses = agent.update(data.sample(8))
assert all(np.isfinite(float(v)) for v in losses.values())
goal = arrays['next_observations'][0]
if width == 8:
    # Archived psm_norm has an undefined norm gradient at an exactly-zero feature.
    # Preserve this reproducer; the adapter must reject its nonfinite inference.
    assert not np.isfinite(np.asarray(agent.measure.params['xphi_out']['kernel'])).all()
    try:
        infer_task(agent, data, goal, 4)
    except FloatingPointError:
        print(json.dumps({'degenerate_feature_rejected': True}))
        sys.exit(0)
    raise AssertionError('Nonfinite inferred coordinate was accepted')
task, diagnostic = infer_task(agent, data, goal, 4)
repeat, _ = infer_task(agent, data, goal, 4)
np.testing.assert_array_equal(task.w_inf, repeat.w_inf)
np.testing.assert_array_equal(task.sample_actions(arrays['observations'][:2]),
                              repeat.sample_actions(arrays['observations'][:2]))
print(json.dumps(diagnostic))
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=code,
                            env={**os.environ, 'PYTHONPATH': str(code), 'JAX_PLATFORMS': 'cpu',
                                 'OMP_NUM_THREADS': '2', 'AFFINE_TEST_WIDTH': str(width)},
                            text=True, capture_output=True, timeout=120, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    diagnostic = json.loads(result.stdout.splitlines()[-1])
    if width == 8:
        assert diagnostic['degenerate_feature_rejected']
        return
    assert diagnostic['actor_updates'] == 2
    assert diagnostic['coordinate_updates'] == 3
    assert diagnostic['representation_unchanged']
    assert diagnostic['training_rng_unchanged']
