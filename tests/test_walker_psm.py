"""Catch ExORL row-shifts, reward leakage and unverifiable snapshot execution."""

import importlib

import numpy as np
import pytest


def api():
    return importlib.import_module('tools.walker_psm.data')


def episode():
    return {
        'observation': np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], np.float32),
        'action': np.array([[0.0], [0.2], [0.7]], np.float32),
        'discount': np.ones((3, 1), np.float32),
        'physics': np.array([[100.0, 101.0], [200.0, 201.0], [300.0, 301.0]], np.float64),
        'reward': np.array([[999.0], [999.0], [999.0]], np.float32),
    }


def test_dummy_action_is_excluded_and_next_physics_aligns():
    out = api().episode_transitions(episode())
    np.testing.assert_array_equal(out['observations'], [[10, 11], [20, 21]])
    np.testing.assert_allclose(out['actions'], [[0.2], [0.7]])
    np.testing.assert_array_equal(out['next_observations'], [[20, 21], [30, 31]])
    np.testing.assert_array_equal(out['next_physics'], [[200, 201], [300, 301]])
    assert 'rewards' not in out
    assert 'reward' not in out


def test_nonunit_environment_discount_cannot_silently_bootstrap():
    ep = episode()
    ep['discount'][-1] = 0
    with pytest.raises(ValueError, match='discount'):
        api().episode_transitions(ep)


def test_episode_order_is_numeric_and_row_ids_survive_sampling(tmp_path):
    np.savez(tmp_path / 'episode_10_2.npz', **episode())
    ep = episode()
    ep['observation'] += 100
    np.savez(tmp_path / 'episode_2_2.npz', **ep)
    ds, manifest = api().load_episodes(tmp_path, max_episodes=2, expected_length=2)
    np.testing.assert_array_equal(ds['observations'][:, 0], [110, 120, 10, 20])
    batch = api().sample_batch(ds, np.array([3, 0, 3]))
    np.testing.assert_array_equal(batch['index'], [3, 0, 3])
    assert set(batch) == {'observations', 'actions', 'next_observations', 'index'}
    assert manifest['transitions'] == 4
    assert len(manifest['episodes']) == 2


def test_incomplete_requested_buffer_fails_closed(tmp_path):
    np.savez(tmp_path / 'episode_0_2.npz', **episode())
    with pytest.raises(ValueError, match='episodes'):
        api().load_episodes(tmp_path, max_episodes=2, expected_length=2)


def test_source_hash_guard_rejects_changed_code(tmp_path):
    import json

    from tools.walker_psm.run import verify_snapshot

    path = tmp_path / 'a.py'
    path.write_text('a = 1\n')
    (tmp_path / 'source_manifest.json').write_text(json.dumps({'files': {'a.py': api().sha256(path)}}))
    verify_snapshot(tmp_path)
    path.write_text('a = 2\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_snapshot(tmp_path)


def test_checkpoint_restores_optimizer_and_sampling_stream(tmp_path):
    import flax.struct
    import jax.numpy as jnp

    from tools.walker_psm.run import restore_checkpoint, save_checkpoint

    @flax.struct.dataclass
    class State:
        params: object
        opt_state: object

    state = State(jnp.array([1.0, 2.0]), {'moment': jnp.array([0.25, 0.5])})
    rng = np.random.default_rng(13)
    path = tmp_path / 'checkpoint.pkl'
    save_checkpoint(path, state, 42, rng, 'abc')
    restored, step, restored_rng = restore_checkpoint(path, state, 'abc')
    assert step == 42
    np.testing.assert_array_equal(restored.opt_state['moment'], [0.25, 0.5])
    np.testing.assert_array_equal(rng.integers(100, size=10), restored_rng.integers(100, size=10))
    with pytest.raises(ValueError, match='flags'):
        restore_checkpoint(path, state, 'wrong')


def test_seed_statistics_average_tasks_before_seed_interval():
    from tools.walker_psm.report import seed_statistics

    report = seed_statistics([[0, 100], [20, 80], [40, 60]])
    assert report['seed_task_means'] == [50.0, 50.0, 50.0]
    assert report['mean'] == 50.0
    assert report['ci95_halfwidth'] == 0.0


def test_bounded_sampler_cannot_accept_released_invalid_actions(tmp_path):
    from tools.walker_psm.run import validate_proto_table

    table = np.full((85536, 6), -1.5, np.float32)
    path = tmp_path / 'invalid.npy'
    np.save(path, table)
    with pytest.raises(ValueError, match='range'):
        validate_proto_table(path, table, 'bounded')


@pytest.mark.parametrize('changed', ['source_manifest', 'dataset_manifest'])
def test_changed_sidecar_cannot_rebind_checkpoint_provenance(tmp_path, changed):
    from tools.walker_psm.data import verify_run_sidecars

    flags = {}
    for name in ('source_manifest', 'dataset_manifest'):
        path = tmp_path / f'{name}.json'
        path.write_text('{"original": true}\n')
        flags[f'{name}_sha256'] = api().sha256(path)
    verify_run_sidecars(tmp_path, flags)
    (tmp_path / f'{changed}.json').write_text('{"replacement": true}\n')
    with pytest.raises(ValueError, match=changed):
        verify_run_sidecars(tmp_path, flags)
