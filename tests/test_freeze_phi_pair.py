"""Contracts for the matched freeze-phi continuation driver."""

import copy

import numpy as np
import pytest

import utils.xla_guard  # noqa: F401  -- MUST precede jax imports in the driver under test
from tools.train_freeze_phi_pair import (
    build_budget,
    build_pair_configs,
    create_output_tree,
    preserve_numpy_rng,
    update_pair,
    validate_preimage_pairing,
    validate_source_flags,
)


def _source_flags():
    return {
        'seed': 0,
        'env_name': 'antmaze-medium-navigate-singletask-v0',
        'online_steps': 0,
        'dataset_fraction': 1.0,
        'dataset_fraction_seed': 0,
        'balanced_sampling': 0,
        'agent': {
            'agent_name': 'psmflow',
            'measure_action_input': 'action',
            'psi_form': 'affine',
            'policy_index': 'latent',
            'acting': 'gpi',
            'index_agg': 'max',
            'gpi_select': 'argmax',
            'discount': 0.99,
            'train_actor': False,
            'flow_ckpt_path': '/flow/source',
            'flow_ckpt_epoch': 500000,
            'preimage_path': '/data/preimages.npz',
            'use_point_preimage': True,
            'measure_u_samples': 1,
            'dsrl_na': {'enabled': False},
            'action_critic': {'enabled': False},
            'arbitrary_preserved_setting': {'x': [1, 2, 3]},
        },
    }


def test_budget_uses_absolute_checkpoint_steps_and_always_includes_final():
    budget = build_budget(source_updates=50_000, additional_steps=450_123, save_interval=50_000)

    assert budget.total_updates == 500_123
    assert budget.save_steps == (
        50_000, 100_000, 150_000, 200_000, 250_000, 300_000,
        350_000, 400_000, 450_000, 500_000, 500_123,
    )


def test_smoke_budget_finishes_at_50200_not_500k_or_550k():
    budget = build_budget(source_updates=50_000, additional_steps=200, save_interval=200)
    assert budget.total_updates == 50_200
    assert budget.save_steps == (50_000, 50_200)


@pytest.mark.parametrize(
    ('path', 'value', 'message'),
    [
        (('seed',), 1, 'seed=0'),
        (('env_name',), 'cube-single-play-singletask-v0', 'AntMaze'),
        (('agent', 'measure_action_input'), 'latent', 'action-conditioned'),
        (('agent', 'psi_form'), 'free', 'affine'),
        (('agent', 'discount'), 0.98, '0.99'),
        (('agent', 'train_actor'), True, 'actor-free'),
        (('agent', 'dsrl_na', 'enabled'), True, 'DSRL'),
        (('agent', 'action_critic', 'enabled'), True, 'action critic'),
    ],
)
def test_source_contract_rejects_confounded_runs(path, value, message):
    flags = _source_flags()
    node = flags
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        validate_source_flags(flags)


def test_pair_configs_preserve_every_source_setting_and_differ_only_in_train_phi():
    source = _source_flags()['agent']
    original = copy.deepcopy(source)

    control, frozen = build_pair_configs(source)

    assert source == original
    assert control['train_phi'] is True
    assert frozen['train_phi'] is False
    control_without_switch = {k: v for k, v in control.items() if k != 'train_phi'}
    frozen_without_switch = {k: v for k, v in frozen.items() if k != 'train_phi'}
    assert control_without_switch == original
    assert frozen_without_switch == original


def test_preimage_pairing_requires_exact_metadata_and_dataset_rows(tmp_path):
    flow = tmp_path / 'flow'
    flow.mkdir()
    observations = np.arange(30, dtype=np.float32).reshape(10, 3)
    meta = {
        'env_name': 'antmaze-medium-navigate-singletask-v0',
        'restore_path': str(flow),
        'restore_epoch': 500000,
        'dataset_fraction': 1.0,
        'dataset_fraction_seed': 0,
        'sampled_batch': False,
    }

    validate_preimage_pairing(
        meta,
        env_name=meta['env_name'],
        flow_ckpt_path=str(flow),
        flow_ckpt_epoch=500000,
        dataset_fraction=1.0,
        dataset_fraction_seed=0,
        augmented_observations=observations.copy(),
        dataset_observations=observations,
    )

    wrong = observations.copy()
    wrong[-1, -1] += 1
    with pytest.raises(ValueError, match='observations differ'):
        validate_preimage_pairing(
            meta,
            env_name=meta['env_name'],
            flow_ckpt_path=str(flow),
            flow_ckpt_epoch=500000,
            dataset_fraction=1.0,
            dataset_fraction_seed=0,
            augmented_observations=wrong,
            dataset_observations=observations,
        )


def test_full_dataset_allows_legacy_sidecar_without_fraction_keys(tmp_path):
    flow = tmp_path / 'flow'
    flow.mkdir()
    observations = np.arange(12, dtype=np.float32).reshape(4, 3)
    meta = {
        'env_name': 'antmaze-medium-navigate-singletask-v0',
        'restore_path': str(flow),
        'restore_epoch': 500000,
    }

    validate_preimage_pairing(
        meta,
        env_name=meta['env_name'],
        flow_ckpt_path=str(flow),
        flow_ckpt_epoch=500000,
        dataset_fraction=1.0,
        dataset_fraction_seed=0,
        augmented_observations=observations,
        dataset_observations=observations.copy(),
    )

    with pytest.raises(ValueError, match='predates dataset_fraction'):
        validate_preimage_pairing(
            meta,
            env_name=meta['env_name'],
            flow_ckpt_path=str(flow),
            flow_ckpt_epoch=500000,
            dataset_fraction=0.5,
            dataset_fraction_seed=1,
            augmented_observations=observations,
            dataset_observations=observations.copy(),
        )


def test_output_root_must_be_new(tmp_path):
    root = tmp_path / 'pair'
    paths = create_output_tree(root)
    assert paths == {'control': root / 'control', 'frozen': root / 'frozen'}
    assert all(path.is_dir() for path in paths.values())

    with pytest.raises(FileExistsError, match='new output root'):
        create_output_tree(root)


def test_eval_rng_context_restores_training_batch_sequence():
    np.random.seed(17)
    expected_first = np.random.randint(0, 1_000_000, size=8)
    expected_second = np.random.randint(0, 1_000_000, size=8)

    np.random.seed(17)
    actual_first = np.random.randint(0, 1_000_000, size=8)
    with preserve_numpy_rng(seed=999):
        _ = np.random.randint(0, 1_000_000, size=100)
    actual_second = np.random.randint(0, 1_000_000, size=8)

    np.testing.assert_array_equal(actual_first, expected_first)
    np.testing.assert_array_equal(actual_second, expected_second)


def test_update_pair_passes_the_same_batch_object_to_both_arms():
    seen = []

    class FakeAgent:
        def __init__(self, arm, rng=7):
            self.arm = arm
            self.rng = np.asarray(rng)

        def update(self, batch):
            seen.append((self.arm, id(batch)))
            return FakeAgent(self.arm, int(self.rng) + 1), {'loss': float(self.rng)}

    batch = {'observations': np.ones((2, 3), dtype=np.float32)}
    control, frozen, infos = update_pair(FakeAgent('control'), FakeAgent('frozen'), batch)

    assert seen == [('control', id(batch)), ('frozen', id(batch))]
    assert int(control.rng) == int(frozen.rng) == 8
    assert set(infos) == {'control', 'frozen'}


def test_update_pair_refuses_diverged_rng_streams():
    class FakeAgent:
        def __init__(self, rng):
            self.rng = np.asarray(rng)

        def update(self, batch):
            return self, {}

    with pytest.raises(AssertionError, match='RNG streams diverged'):
        update_pair(FakeAgent(1), FakeAgent(2), {})
