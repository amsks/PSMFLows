"""Evaluation-only regression: fixed panels must become fresh, reproducible draws."""

import copy
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tools.walker_psm.eval_identity import FreshDrawsAgent


@pytest.fixture
def agent():
    # Use the actual checkpoint-producing sampler, with small networks for CPU tests.
    path = Path(__file__).resolve().parents[1] / 'outputs/walker_identity_20260913/code/identity_psm_agent.py'
    if not path.is_file():
        pytest.skip('Walker identity campaign snapshot is not installed')
    spec = importlib.util.spec_from_file_location('walker_identity_regression_agent', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = {
        'psi_form': 'affine', 'z_dim': 2, 'num_parallel': 2, 'norm_z': True,
        'gpi_num_u': 4, 'lr_phi': 1e-4, 'lr_psi': 1e-4,
        'actor_pessimism_penalty': 0.5,
        'phi': {'hidden_dim': 8, 'hidden_layers': 1},
        'sf': {'hidden_dim': 8, 'embedding_layers': 2, 'hidden_layers': 1},
        'affine': {'w_dim': 2, 'encoder_hidden': 8, 'encoder_layers': 1, 'norm_w': True},
    }
    return module.IdentityPSMAgent.create(0, jnp.zeros((1, 24)), jnp.zeros((1, 6)), cfg)


def evaluation_policy(agent, seed=0):
    return FreshDrawsAgent(agent, eval_seed=seed)


def action_sequence(policy):
    observations = jnp.zeros((4, 24))
    return np.stack([np.asarray(policy.sample_actions(observations)) for _ in range(3)])


def test_evaluation_redraws_actions_at_identical_states(agent):
    """Missing/unchanged explicit RNG keys must fail this test."""
    actions = action_sequence(evaluation_policy(agent))
    assert not np.array_equal(actions[0], actions[1]), 'GPI reused the same action candidate panel'
    assert not np.array_equal(actions[1], actions[2])


def test_evaluation_seed_reproduces_complete_action_sequence(agent):
    first = action_sequence(evaluation_policy(agent, seed=19))
    second = action_sequence(evaluation_policy(agent, seed=19))
    np.testing.assert_array_equal(first, second)


def test_evaluation_does_not_mutate_checkpoint_state(agent):
    before = [np.asarray(x).copy() for x in jax.tree_util.tree_leaves(agent)]
    action_sequence(evaluation_policy(agent))
    for old, new in zip(before, jax.tree_util.tree_leaves(agent), strict=True):
        np.testing.assert_array_equal(old, np.asarray(new))


def test_different_evaluation_seed_changes_actions(agent):
    assert not np.array_equal(
        action_sequence(evaluation_policy(agent, seed=0)), action_sequence(evaluation_policy(agent, seed=1))
    )


def test_task_inference_resets_eval_stream_without_mutating_base(agent):
    policy = evaluation_policy(agent, seed=11)
    observations, rewards = jnp.ones((8, 24)), jnp.ones(8)
    first = policy.infer_eval_z(observations, rewards)
    second = policy.infer_eval_z(observations, rewards)
    np.testing.assert_array_equal(action_sequence(first), action_sequence(second))
    assert [t['calls'] for t in policy.rng_audit()] == [3, 3]
    assert policy.rng_audit()[0]['key_sha256'] == policy.rng_audit()[1]['key_sha256']
    np.testing.assert_array_equal(agent.task_z, np.zeros(2))


def report_fixture():
    from tools.walker_psm.eval_identity import RNG_MODE

    provenance = {'seed': 0, 'run_dir': '/example/seed0', 'flags_sha256': 'f' * 64, 'checkpoint_sha256': 'c' * 64}
    manifest = {
        'protocol': {'checkpoint_step': 2000000, 'eval_seed': 0, 'eval_workers': 4, 'inference_samples': 10000},
        'evaluator_sha256': 'e' * 64,
        'verified_checkpoint_provenance': [provenance],
        'saved_agent_config': {'psi_form': 'affine'},
    }
    report = {
        'step': 2000000, 'seed': 0, 'run_dir': '/example/seed0', 'flags_sha256': 'f' * 64,
        'input_provenance': copy.deepcopy(provenance), 'evaluator_sha256': 'e' * 64,
        'agent_config': {'psi_form': 'affine'}, 'evaluated_restored_checkpoint': True,
        'eval_seed': 0, 'eval_workers': 4, 'inference_samples': 10000,
        'inference_index_sha256': 'i' * 64, 'rng_mode': RNG_MODE,
        'state_sha256_before': 'b' * 64, 'state_sha256_after': 'b' * 64,
        'rng_audit': [
            {'calls': 1000, 'first_keys': [[1, 2], [3, 4]], 'key_sha256': 'a' * 64} for _ in range(4)
        ],
        'task_mean': 2.5,
        'tasks': {
            name: {'mean_return': value, 'episode_returns': [value] * 4, 'episode_lengths': [1000] * 4,
                   'episodes': 4, 'task_z': [0.0] * 128}
            for name, value in zip(('stand', 'walk', 'run', 'flip'), (1.0, 2.0, 3.0, 4.0), strict=True)
        },
    }
    return manifest, report


def test_report_gate_accepts_verified_complete_evaluation():
    from tools.walker_psm import eval_identity

    assert callable(getattr(eval_identity, 'validate_report', None)), 'Missing corrected-evaluation report gate'
    manifest, report = report_fixture()
    assert eval_identity.validate_report(report, manifest, seed=0, episodes=4) == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize('fault', [
    'checkpoint', 'flags', 'step', 'seed', 'missing_task', 'episode_count', 'horizon', 'nan_return',
    'wrong_mean', 'fixed_rng', 'repeated_keys', 'wrong_call_count', 'changed_state', 'not_restored', 'workers',
])
def test_report_gate_rejects_incomparable_or_invalid_results(fault):
    from tools.walker_psm import eval_identity

    assert callable(getattr(eval_identity, 'validate_report', None)), 'Missing corrected-evaluation report gate'
    manifest, report = report_fixture()
    if fault == 'checkpoint':
        report['input_provenance']['checkpoint_sha256'] = 'wrong'
    elif fault == 'flags':
        report['flags_sha256'] = 'wrong'
    elif fault in ('step', 'seed'):
        report[fault] += 1
    elif fault == 'missing_task':
        del report['tasks']['flip']
    elif fault == 'episode_count':
        report['tasks']['stand']['episodes'] = 500
    elif fault == 'horizon':
        report['tasks']['stand']['episode_lengths'][0] = 999
    elif fault == 'nan_return':
        report['tasks']['stand']['episode_returns'][0] = float('nan')
    elif fault == 'wrong_mean':
        report['tasks']['stand']['mean_return'] += 1
    elif fault == 'fixed_rng':
        report['rng_mode'] = 'fixed'
    elif fault == 'repeated_keys':
        report['rng_audit'][0]['first_keys'] = [[1, 2], [1, 2]]
    elif fault == 'wrong_call_count':
        report['rng_audit'][0]['calls'] -= 1
    elif fault == 'changed_state':
        report['state_sha256_after'] = 'd' * 64
    elif fault == 'not_restored':
        report['evaluated_restored_checkpoint'] = False
    elif fault == 'workers':
        report['eval_workers'] = 1
    with pytest.raises(ValueError):
        eval_identity.validate_report(report, manifest, seed=0, episodes=4)


def test_report_gate_rejects_unregistered_training_seed():
    from tools.walker_psm.eval_identity import validate_report

    manifest, report = report_fixture()
    report['seed'] = 3
    with pytest.raises(ValueError, match='seed'):
        validate_report(report, manifest, seed=3, episodes=4)


@pytest.fixture
def final_reports(tmp_path):
    from tools.walker_psm.eval_identity import sha256

    manifest, original = report_fixture()
    manifest['protocol']['episodes_per_task'] = 500
    manifest['verified_checkpoint_provenance'] = []
    paths = []
    for seed in range(3):
        report = copy.deepcopy(original)
        report['seed'] = seed
        report['run_dir'] = f'/example/seed{seed}'
        report['input_provenance'].update(seed=seed, run_dir=report['run_dir'])
        manifest['verified_checkpoint_provenance'].append(copy.deepcopy(report['input_provenance']))
        for task_number, task in enumerate(report['tasks'].values()):
            value = float(5 * seed + 10 * task_number)
            task.update(mean_return=value, episode_returns=[value] * 500,
                        episode_lengths=[1000] * 500, episodes=500)
        report['task_mean'] = float(15 + 5 * seed)
        for trace in report['rng_audit']:
            trace['calls'] = 125000
        path = tmp_path / f'seed{seed}.json'
        path.write_text(json.dumps(report))
        path.with_suffix('.validated.json').write_text(json.dumps({
            'report_sha256': sha256(path), 'all_checks_passed': True,
        }))
        paths.append(path)
    return manifest, paths


def test_final_aggregate_averages_tasks_before_seed_uncertainty(final_reports):
    from tools.walker_psm.eval_identity import aggregate_reports

    manifest, paths = final_reports
    result = aggregate_reports(manifest, paths)
    assert result['seed_task_means'] == [15.0, 20.0, 25.0]
    assert result['mean'] == 20.0
    assert result['ci95_halfwidth'] == pytest.approx(12.420688558597728)


def test_final_aggregate_rejects_duplicate_report_paths(final_reports):
    from tools.walker_psm.eval_identity import aggregate_reports

    manifest, paths = final_reports
    with pytest.raises(ValueError, match='distinct'):
        aggregate_reports(manifest, [paths[0], paths[0], paths[2]])


def test_final_aggregate_rejects_modified_report_after_validation(final_reports):
    from tools.walker_psm.eval_identity import aggregate_reports

    manifest, paths = final_reports
    paths[0].write_text(paths[0].read_text() + '\n')
    with pytest.raises(ValueError, match='receipt'):
        aggregate_reports(manifest, paths)
