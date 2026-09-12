import json

import pytest

from tools.report_seed_comparison import build_report


def _write_run(tmp_path, seed, task, success, *, eval_seed=0, step=500000, dsrl_na=None):
    run = tmp_path / f'run_{seed}'
    run.mkdir(exist_ok=True)
    flags = {
        'seed': seed,
        'env_name': f'cube-single-play-singletask-task{task}-v0',
        'offline_steps': step,
        'restore_path': None,
        'restore_epoch': None,
        'agent': {
            'psi_form': 'affine', 'policy_index': 'latent', 'acting': 'gpi',
            'train_actor': False, 'flow_ckpt_path': '/flow',
            'flow_ckpt_epoch': 500000, 'preimage_path': '/preimage',
            'use_point_preimage': True,
            **({'dsrl_na': dsrl_na} if dsrl_na is not None else {}),
        },
    }
    (run / 'flags.json').write_text(json.dumps(flags))
    report = run / f'eval_{task}_{eval_seed}_{step}.json'
    report.write_text(json.dumps({
        'env': f'cube-single-play-singletask-task{task}-v0', 'seed': eval_seed, 'restore_path': str(run),
        'restore_epoch': step, 'num_episodes': 500, 'num_success': success,
        'success': success / 500, 'flow_ckpt_path': '/flow',
        'preimage_path': '/preimage', 'agent_config_source': {
            'flags_json': str(run / 'flags.json'), 'cli_overrides': [
                'flow_ckpt_epoch', 'flow_ckpt_path', 'preimage_path', 'use_point_preimage'
            ]},
    }))
    return report


def _manifest(tmp_path, reports):
    return {'label': 'test', 'expected_seeds': [0, 1, 2], 'restore_epoch': 500000,
            'num_episodes': 500,
            'tasks': {str(t): f'cube-single-play-singletask-task{t}-v0' for t in range(1, 6)},
            'reports': [{'seed': s, 'task': t, 'path': str(p)} for (s, t), p in reports.items()]}


def test_averages_tasks_per_seed_then_student_ci(tmp_path):
    reports = {(s, t): _write_run(tmp_path, s, t, 100 + 10 * s + t)
               for s in range(3) for t in range(1, 6)}
    out = build_report(_manifest(tmp_path, reports))
    assert out['seed_scores'] == pytest.approx([0.206, 0.226, 0.246])
    assert out['mean'] == pytest.approx(0.226)
    assert out['n_seeds'] == 3
    assert out['zero_shot'] is True
    assert out['ci95'] == pytest.approx(4.302652729911275 * 0.02 / (3 ** 0.5))


def test_rejects_incomplete_or_duplicate_matrix(tmp_path):
    reports = {(s, 1): _write_run(tmp_path, s, 1, 1) for s in range(3)}
    with pytest.raises(ValueError, match='complete'):
        build_report(_manifest(tmp_path, reports))
    reports[(0, 2)] = reports[(0, 1)]
    with pytest.raises(ValueError, match='unique'):
        build_report(_manifest(tmp_path, reports))


def test_requires_five_matching_domain_tasks(tmp_path):
    manifest = _manifest(tmp_path, {})
    manifest['tasks'] = {str(t): f'cube-single-play-singletask-task{t}-v0' for t in range(1, 5)}
    with pytest.raises(ValueError, match='exactly'):
        build_report(manifest)
    manifest['tasks'] = {str(t): 'cube-single-play-singletask-task1-v0' for t in range(1, 6)}
    with pytest.raises(ValueError, match='matching'):
        build_report(manifest)
def test_uses_training_seed_and_rejects_nonfinal_or_policy_override(tmp_path):
    reports = {(s, t): _write_run(tmp_path, s, t, 1, eval_seed=99, step=500000)
               for s in range(3) for t in range(1, 6)}
    out = build_report(_manifest(tmp_path, reports))
    assert out['training_seeds'] == [0, 1, 2]
    reports[(0, 1)] = _write_run(tmp_path, 0, 1, 1, step=250000)
    with pytest.raises(ValueError, match='restore_epoch'):
        build_report(_manifest(tmp_path, reports))

    reports = {(s, t): _write_run(tmp_path, s, t, 1) for s in range(3) for t in range(1, 6)}
    d = json.loads(reports[(0, 1)].read_text())
    d['agent_config_source']['flags_json'] = str(tmp_path / 'run_1' / 'flags.json')
    reports[(0, 1)].write_text(json.dumps(d))
    with pytest.raises(ValueError, match='outside report restore_path'):
        build_report(_manifest(tmp_path, reports))

    reports[(0, 1)] = _write_run(tmp_path, 0, 1, 1)
    d = json.loads(reports[(0, 1)].read_text())
    d['agent_config_source']['cli_overrides'].append('gpi_num_u')
    reports[(0, 1)].write_text(json.dumps(d))
    with pytest.raises(ValueError, match='policy-changing'):
        build_report(_manifest(tmp_path, reports))

    reports = {(s, t): _write_run(tmp_path, s, t, 1) for s in range(3) for t in range(1, 6)}
    flags = reports[(1, 1)].parent / 'flags.json'
    cfg = json.loads(flags.read_text())
    cfg['agent']['flow_ckpt_epoch'] = 400000
    flags.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='relevant agent config'):
        build_report(_manifest(tmp_path, reports))

    reports = {(s, t): _write_run(tmp_path, s, t, 1) for s in range(3) for t in range(1, 6)}
    cfg = json.loads((reports[(1, 1)].parent / 'flags.json').read_text())
    cfg['online_steps'] = 1
    (reports[(1, 1)].parent / 'flags.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='online continuation'):
        build_report(_manifest(tmp_path, reports))

    reports = {(s, t): _write_run(tmp_path, s, t, 1) for s in range(3) for t in range(1, 6)}
    d = json.loads(reports[(0, 1)].read_text())
    d['num_episodes'] = 499
    reports[(0, 1)].write_text(json.dumps(d))
    with pytest.raises(ValueError, match='num_episodes'):
        build_report(_manifest(tmp_path, reports))


def test_real_reward_dsrl_is_not_zero_shot(tmp_path):
    reports = {(s, t): _write_run(tmp_path, s, t, 1, dsrl_na={'enabled': True, 'reward_source': 'real'})
               for s in range(3) for t in range(1, 6)}
    assert build_report(_manifest(tmp_path, reports))['zero_shot'] is False

    reports = {(s, t): _write_run(tmp_path, s, t, 1, dsrl_na={'enabled': True})
               for s in range(3) for t in range(1, 6)}
    assert build_report(_manifest(tmp_path, reports))['zero_shot'] is False
