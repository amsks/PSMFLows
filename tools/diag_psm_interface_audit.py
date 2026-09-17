"""Reproduce algebra checks and revalidate existing freeze-phi reports; no training.

The projected-TD example is a counterexample to a general stability implication,
not a simulation or a diagnosis of the trained PSMFlow checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# isort: off
import utils.xla_guard  # noqa: F401 -- must precede JAX
import jax
import jax.numpy as jnp
import numpy as np

from utils.psm_common import contrastive_loss, off_diagonal_mask, targets_uncertainty
# isort: on


def _json(path):
    return json.loads(path.read_text())


def check_loss():
    """Compare the actual helper and its gradients to independently expanded sums."""
    rng = np.random.default_rng(13)
    p, n, gamma = 2, 7, 0.98
    m = rng.normal(size=(p, n, n)).astype(np.float32)
    target = rng.normal(size=(n, n)).astype(np.float32)
    off, denom = off_diagonal_mask(n)

    def loss(mm, tt):
        return contrastive_loss(mm, jax.lax.stop_gradient(tt), gamma, off, denom)[0]

    got, (dm, dt) = jax.value_and_grad(loss, argnums=(0, 1))(jnp.asarray(m), jnp.asarray(target))
    diff = m.astype(np.float64) - gamma * target.astype(np.float64)
    expected = sum(
        0.5 * sum(diff[k, i, j] ** 2 for i in range(n) for j in range(n) if i != j)
        / (n * (n - 1)) - sum(diff[k, i, i] for i in range(n)) / n
        for k in range(p)
    )
    expected_grad = diff / (n * (n - 1))
    expected_grad[:, np.arange(n), np.arange(n)] = -1 / n
    np.testing.assert_allclose(float(got), expected, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(dm), expected_grad, atol=1e-7, rtol=1e-6)
    assert np.all(np.asarray(dt) == 0)

    two = jnp.asarray(rng.normal(size=(2, 19)).astype(np.float32))
    mean, spread = targets_uncertainty(two, 2)
    min_error = float(jnp.max(jnp.abs(mean - 0.5 * spread - two.min(0))))
    assert min_error < 1e-6
    _, one_spread = targets_uncertainty(two[:1], 1)
    return {
        'contrastive_value_absolute_error': abs(float(got) - expected),
        'contrastive_gradient_max_absolute_error': float(np.max(np.abs(np.asarray(dm) - expected_grad))),
        'target_gradient_exactly_zero': True,
        'two_head_target_vs_elementwise_min_max_error': min_error,
        'single_head_spread_all_nan': bool(np.isnan(np.asarray(one_spread)).all()),
        'single_head_note': 'Existing helper divides by P*(P-1). Default P=2 is unaffected; P=1 is invalid.',
    }


def projected_td_counterexample():
    """Exact finite example and Gaussian quadrature check of its continuous analogue."""
    b = math.sqrt(2) - 1
    gamma = 0.9
    h = np.array([[1.0, -b], [b, -1.0]])  # rows = current action, columns = policy index
    design = np.stack((np.ones(4), h.reshape(-1)), axis=1)
    boot_h = np.tile(np.diag(h), (2, 1)).reshape(-1)  # next action equals the policy index
    boot_design = np.stack((np.ones(4), boot_h), axis=1)
    gram = design.T @ design / 4
    cross = design.T @ boot_design / 4
    transition = gamma * np.linalg.solve(gram, cross)
    forcing = np.linalg.solve(gram, design.mean(0))
    analytic_gain = gamma * (1 + math.sqrt(2)) / 2
    np.testing.assert_allclose(transition, np.diag([gamma, analytic_gain]), atol=1e-14)
    fixed = np.array([1 / (1 - gamma), 0.0])
    np.testing.assert_allclose(forcing + transition @ fixed, fixed, atol=1e-14)
    theta = fixed + np.array([0.0, 1e-3])
    start_error = np.linalg.norm(theta - fixed)
    for _ in range(100):
        theta = forcing + transition @ theta
    growth = np.linalg.norm(theta - fixed) / start_error
    assert growth > 1000

    # G(u)=u, u,c independent N(0,1), f=tanh, h=lambda*f(u)+kappa*f(c).
    # The two-point table above has exactly the same projected matrix.
    nodes, weights = np.polynomial.hermite.hermgauss(48)
    f = np.tanh(math.sqrt(2) * nodes)
    weights /= math.sqrt(math.pi)
    lam, kap = (1 - b) / 2, (1 + b) / 2
    h_cont = lam * f[:, None] + kap * f[None, :]
    joint_weights = weights[:, None] * weights[None, :]
    continuum_gain = gamma * np.sum(joint_weights * h_cont * f[None, :]) / np.sum(
        joint_weights * h_cont ** 2)
    np.testing.assert_allclose(continuum_gain, analytic_gain, atol=1e-14)
    return {
        'discount': gamma,
        'h_action_by_index': h.tolist(),
        'train_action_index_probability': np.full((2, 2), 0.25).tolist(),
        'bootstrap_action_index_probability': (np.eye(2) * 0.5).tolist(),
        'marginal_action_density_ratio': 1.0,
        'finite_joint_max_density_ratio': 2.0,
        'projected_update_matrix': transition.tolist(),
        'projected_spectral_radius': float(np.max(np.abs(np.linalg.eigvals(transition)))),
        'gaussian_quadrature_gain': float(continuum_gain),
        'parameter_error_growth_after_100_fitted_updates': float(growth),
        'true_constant_successor_density': float(fixed[0]),
        'true_solution_in_model_class': True,
        'phi_fixed_and_exact': True,
        'pessimism': 0.0,
        'interpretation': 'Marginal C=1 does not imply projected-TD stability. This is not a live-network measurement.',
    }


def check_minimum():
    rng = np.random.default_rng(31)
    left, right = rng.normal(size=(2, 10000)), rng.normal(size=(2, 10000))
    gamma = 0.99
    violation = np.max(
        gamma * np.abs(left.min(0) - right.min(0)) - gamma * np.max(np.abs(left - right), axis=0))
    assert violation < 1e-12
    density_heads = np.array([[2.0, 0.0], [0.0, 2.0]])
    minimum = density_heads.min(0)
    negative_reward = -np.ones(2)
    return {
        'exact_minimum_backup_sup_norm_lipschitz_bound': gamma,
        'random_check_max_bound_violation': float(violation),
        'density_heads': density_heads.tolist(),
        'head_total_masses': density_heads.sum(1).tolist(),
        'minimum_total_mass': float(minimum.sum()),
        'head_values_at_negative_constant_reward': (density_heads @ negative_reward).tolist(),
        'minimum_value_at_negative_constant_reward': float(minimum @ negative_reward),
        'interpretation': 'The exact min backup is contractive; projection may not be. Density min loses mass and is not pessimistic for every signed reward.',
    }


def check_auxiliary_reward_channels():
    """Test information flow in D2 and the held-readout diagnostic with small real agents."""
    from agents.psmflow import PSMFlowAgent, get_config

    config = get_config()
    with config.unlocked():
        config.allow_untrained_flow = True
        config.z_dim = 4
        config.train_actor = True
        config.acting = 'actor'
        config.actor_mode = 'dsrl_sac'
        config.actor.bc_coeff = 0.0
        for name in ('phi', 'sf', 'actor', 'q_dist', 'dsrl_na', 'action_critic', 'residual'):
            config[name].hidden_dim = 8
            config[name].hidden_layers = 1
        config.actor.vf_hidden_dim = 8
        config.actor.vf_hidden_layers = 1
        config.affine.w_dim = 4
        config.affine.encoder_hidden = 8
        config.affine.encoder_layers = 1
        config.flow.hidden_dims = (8,)
        config.flow.value_hidden_dims = (8,)
        config.dsrl_na.enabled = True
        config.dsrl_na.inner_steps = 1
        config.dsrl_na.reward_source = 'synthetic_w'
        config.dsrl_na.task_conditioned = True
    rng = np.random.default_rng(41)
    batch = {name: rng.normal(size=(8, width)).astype(np.float32)
             for name, width in (('observations', 5), ('next_observations', 5),
                                 ('actions', 2), ('noise_preimage', 2))}
    batch['rewards'] = -np.ones(8, np.float32)
    batch['masks'] = np.ones(8, np.float32)
    agent = PSMFlowAgent.create(0, batch['observations'][:1], batch['actions'][:1], config)
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(7))
    loss, _ = agent.dsrl_qa_loss(batch, sampled, agent.qa.params)
    reward_loss, _ = agent.dsrl_qa_loss({**batch, 'rewards': batch['rewards'] + 99}, sampled, agent.qa.params)
    mask_loss, _ = agent.dsrl_qa_loss({**batch, 'masks': np.zeros(8, np.float32)}, sampled, agent.qa.params)
    assert float(reward_loss) == float(loss)
    assert abs(float(mask_loss) - float(loss)) > 1e-6

    # Holding w/scale does not hold phi(x)^T w when the basis continues changing.
    fixed_config = get_config()
    fixed_config.update(config.to_dict())
    with fixed_config.unlocked():
        fixed_config.dsrl_na.reward_source = 'phi_readout_fixed'
        fixed_config.dsrl_na.task_conditioned = False
    fixed = PSMFlowAgent.create(0, batch['observations'][:1], batch['actions'][:1], fixed_config)
    fit_rewards = np.array([-1, -1, 0, -1, 0, -1, -1, 0], np.float32)
    fixed, _ = fixed.refit_na_reward(batch['next_observations'], fit_rewards)
    before = np.asarray((fixed.phi(batch['next_observations']) @ fixed.na_rw) * fixed.na_rw_scale)
    changed_params = jax.tree_util.tree_map(lambda p: p + 0.02, fixed.phi.params)
    changed = fixed.replace(phi=fixed.phi.replace(params=changed_params))
    after = np.asarray((changed.phi(batch['next_observations']) @ changed.na_rw) * changed.na_rw_scale)
    assert np.array_equal(np.asarray(fixed.na_rw), np.asarray(changed.na_rw))
    assert float(fixed.na_rw_scale) == float(changed.na_rw_scale)
    assert float(np.max(np.abs(after - before))) > 1e-6
    return {
        'synthetic_w_loss_change_when_real_rewards_change': float(reward_loss - loss),
        'synthetic_w_loss_change_when_masks_change': float(mask_loss - loss),
        'held_readout_reward_max_change_after_basis_perturbation': float(np.max(np.abs(after - before))),
        'held_w_and_scale_unchanged': True,
        'interpretation': 'Synthetic-w TD still reads masks. Held w does not freeze a reward defined through an updating phi.',
    }


def validate_freeze_reports():
    base = ROOT / 'outputs/freeze_phi_20260911'
    completion = _json(base / 'pilot/completion.json')
    assert completion['status'] == 'complete'
    assert completion['source_updates'] + completion['additional_steps'] == 500000
    assert all(completion['checks'].values())
    arms, configs = {}, {}
    for arm in ('control', 'frozen'):
        run = base / 'pilot' / arm
        flags = _json(run / 'flags.json')
        assert flags['agent']['train_phi'] == (arm == 'control')
        configs[arm] = {k: v for k, v in flags['agent'].items() if k != 'train_phi'}
        reports = []
        for task in range(1, 6):
            path = base / 'reports' / f'{arm}_task{task}_500000.json'
            report = _json(path)
            assert report['env'] == f'antmaze-medium-navigate-singletask-task{task}-v0'
            assert report['num_episodes'] == 500 and report['restore_epoch'] == 500000
            assert report['train_seed'] == 0 and report['seed'] == 0 and report['num_workers'] == 4
            assert Path(report['restore_path']).resolve() == run.resolve()
            assert Path(report['agent_config_source']['flags_json']).resolve() == (run / 'flags.json').resolve()
            assert report['agent_config_source']['cli_overrides'] == []
            assert report['success'] == report['num_success'] / 500
            episodes = report['per_episode_success']
            assert len(episodes) == 500 and sum(episodes) == report['num_success']
            reports.append({'task': task, 'success': report['success'], 'num_success': report['num_success'],
                            'path': str(path.relative_to(ROOT)),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        with (run / 'train.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        last = rows[-1]
        arms[arm] = {
            'reports': reports,
            'five_task_mean': sum(r['success'] for r in reports) / 5,
            'final_train_metrics': {k: float(v) for k, v in last.items()
                                    if k in ('step', 'training/psm_loss', 'training/phi_gram_dev',
                                             'training/td_target_absmean', 'training/psi_absmean')},
        }
    assert configs['control'] == configs['frozen']
    source = _json(ROOT / 'outputs/affine_action_20260911/reports/antmaze_sd0_task1_50000.json')
    bc = [_json(ROOT / f'outputs/affine_action_20260911/bc_reports/task{task}.json') for task in range(1, 6)]
    assert all(r['num_episodes'] == 500 for r in bc)
    return {
        'source_task1_50k_success': source['success'],
        'source_updates': completion['source_updates'],
        'additional_updates': completion['additional_steps'],
        'completion_checks': completion['checks'],
        'arms': arms,
        'same_flow_bc_task_successes': [r['success'] for r in bc],
        'same_flow_bc_five_task_mean': sum(r['success'] for r in bc) / 5,
        'interpretation': 'Existing completed single-seed experiment, reports revalidated today; no fresh rollouts or across-seed CI.',
    }


def inspect_preimages(data_root):
    reports = {}
    for name in ('cube-single-play', 'antmaze-medium-navigate'):
        path = data_root / 'preimages' / f'{name}.npz'
        if not path.exists():
            reports[name] = {'path': str(path), 'available': False}
            continue
        with np.load(path, allow_pickle=False) as dataset:
            point = dataset['noise_preimage_point'].astype(np.float64)
            valid = np.isfinite(point).all(1) & ((point ** 2).sum(1) <= 100)
            if 'preimage_valid' in dataset:
                valid &= dataset['preimage_valid'].reshape(-1) > 0.5
            masks = dataset.get('masks')
            rewards = dataset.get('rewards')
            reports[name] = {
                'path': str(path), 'available': True, 'rows': len(point),
                'valid_point_rows': int(valid.sum()),
                'valid_point_rows_clipped_at_u_clip_3': int((np.abs(point[valid]) > 3).any(1).sum()),
                'valid_point_row_clip_fraction': float((np.abs(point[valid]) > 3).any(1).mean()),
                'mask_values_counts': (dict(zip(*[x.tolist() for x in np.unique(masks, return_counts=True)]))
                                       if masks is not None else None),
                'masks_equal_negative_task_rewards': (bool(np.array_equal(masks, -rewards))
                                                       if masks is not None and rewards is not None else None),
                'note': 'Point and stored-mask validity only; no new inversion or decode error evaluation.',
            }
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report_out', type=Path, required=True)
    parser.add_argument('--data_root', type=Path, default=Path('/mnt/home/amohan/psm-data'))
    args = parser.parse_args()
    report = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'kind': 'algebra_checks_and_existing_artifact_audit',
        'loss': check_loss(),
        'projected_td_counterexample': projected_td_counterexample(),
        'minimum_backup': check_minimum(),
        'auxiliary_reward_channels': check_auxiliary_reward_channels(),
        'freeze_phi': validate_freeze_reports(),
        'preimage_arrays': inspect_preimages(args.data_root),
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ('agents/psmflow.py', 'utils/psm_common.py', 'utils/psm_networks.py',
                                       'tools/diag_psm_interface_audit.py', 'tools/diag_fitted_vs_true_return.py')},
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'report_out': str(args.report_out), 'loss': report['loss'],
                      'projected_td_gain': report['projected_td_counterexample']['projected_spectral_radius'],
                      'freeze_phi_means': {k: v['five_task_mean'] for k, v in report['freeze_phi']['arms'].items()},
                      'preimage_arrays': report['preimage_arrays']}, indent=2))


if __name__ == '__main__':
    main()
