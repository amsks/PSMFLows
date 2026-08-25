"""Decode-recovery diagnostics: the chunked k-NN helper and the two-test tool.

The tool's numbers only mean something against a trained flow; what is checked here is
that both tests run end to end off an npz alone (no env), that every latent source and
baseline is populated, and that `knn_indices` agrees exactly with a brute-force search.
"""
import json

import jax
import ml_collections
import numpy as np
import pytest

from agents.fql import FQLAgent, get_config
from utils.flow_inversion import (
    augment_dataset_with_point_preimage,
    augment_dataset_with_preimage_distribution,
    save_augmented_dataset,
)
from utils.nn_search import knn_indices

N, OBS, ACT = 24, 4, 2


@pytest.fixture(autouse=True)
def _force_float32():
    """The BC-flow inversion scan is not x64-safe; see tests/test_flow_inversion.py."""
    prev = jax.config.read('jax_enable_x64')
    jax.config.update('jax_enable_x64', False)
    yield
    jax.config.update('jax_enable_x64', prev)


def _brute(queries, points, k, scale=None):
    q, p = queries, points
    if scale is not None:
        q, p = q / scale, p / scale
    d = np.linalg.norm(q[:, None] - p[None], axis=-1)
    idx = np.argsort(d, axis=1, kind='stable')[:, :k]
    return idx, np.take_along_axis(d, idx, 1)


def test_knn_matches_brute_force():
    rng = np.random.default_rng(0)
    points = rng.standard_normal((250, 3)).astype(np.float32)
    queries = rng.standard_normal((7, 3)).astype(np.float32)

    idx, dist = knn_indices(queries, points, k=5, chunk=37)
    b_idx, b_dist = _brute(queries, points, 5)
    assert np.array_equal(idx, b_idx)
    np.testing.assert_allclose(dist, b_dist, atol=1e-4)


def test_knn_scale_and_exclude():
    rng = np.random.default_rng(1)
    points = rng.standard_normal((120, 3)).astype(np.float32)
    scale = np.array([1.0, 10.0, 100.0], np.float32)

    idx, dist = knn_indices(points[:6], points, k=4, chunk=25, scale=scale)
    b_idx, b_dist = _brute(points[:6], points, 4, scale=scale)
    assert np.array_equal(idx, b_idx)
    np.testing.assert_allclose(dist, b_dist, atol=1e-4)
    assert np.array_equal(idx[:, 0], np.arange(6))  # each query matches itself first

    rows = np.arange(6)
    idx_ex, _ = knn_indices(points[rows], points, k=4, chunk=25, scale=scale, exclude=rows)
    assert not np.any(idx_ex == rows[:, None])
    # Excluding the self-match shifts the brute-force ranking by one.
    assert np.array_equal(idx_ex, _brute(points[rows], points, 5, scale=scale)[0][:, 1:])


@pytest.fixture(scope='module')
def npz(tmp_path_factory):
    """A tiny preimage npz, written by the same precompute path the real ones come from."""
    rng = np.random.default_rng(0)
    ds = dict(
        observations=rng.standard_normal((N, OBS)).astype(np.float32),
        actions=np.clip(rng.standard_normal((N, ACT)), -1, 1).astype(np.float32),
        next_observations=rng.standard_normal((N, OBS)).astype(np.float32),
        terminals=np.zeros((N,), np.float32),
    )
    cfg = get_config()
    cfg['actor_hidden_dims'] = (32, 32)
    cfg['value_hidden_dims'] = (32, 32)
    cfg['flow_steps'] = 3
    agent = FQLAgent.create(0, ds['observations'][:1], ds['actions'][:1], cfg)
    inv = ml_collections.ConfigDict(
        dict(num_clusters=2, alpha=1.0, num_samples=6, n_steps=1, n_initial_steps=2,
             batch_size=8, seed=0))
    aug = augment_dataset_with_preimage_distribution(agent, ds, inv)
    aug = augment_dataset_with_point_preimage(agent, aug, inv)

    path = str(tmp_path_factory.mktemp('decode') / 'pre.npz')
    save_augmented_dataset(path, aug)
    with open(path + '.meta.json', 'w') as f:
        json.dump(dict(env_name='unit-test-v0', restore_path=None, restore_epoch=None,
                       flow_steps=3), f)
    return path


def _cfg(npz_path, **over):
    from omegaconf import OmegaConf
    agent = OmegaConf.create(dict(get_config()))
    agent.actor_hidden_dims = [32, 32]
    agent.value_hidden_dims = [32, 32]
    agent.flow_steps = 3
    base = dict(agent=agent, env_name='unit-test-v0', seed=0, restore_path=None,
                restore_epoch=None, report_out=None, preimage_npz=npz_path,
                preimage_limit=8, inversion=dict(batch_size=8, allow_untrained=True))
    return OmegaConf.create({**base, **over})


def test_tool_runs_both_tests(npz, tmp_path):
    from tools.validate_decode_recovery import main

    out = str(tmp_path / 'report.json')
    cfg = _cfg(npz, report_out=out, t1_sources=['mixture', 'point', 'prior'],
               t2_states=6, t2_latents=3, t2_k=4, t2_chunk=7)
    main.__wrapped__(cfg)

    with open(out) as f:
        report = json.load(f)

    t1 = report['test1_decode_recovery']
    assert t1['n_rows'] == 8
    for source in ('mixture', 'point', 'prior'):
        assert np.isfinite(t1[source]['mse'])
        assert len(t1[source]['max_abs_per_dim']) == ACT
        assert t1[source]['l2']['min'] <= t1[source]['l2']['median'] <= t1[source]['l2']['max']

    t2 = report['test2_buffer_recovery']
    assert (t2['n_states'], t2['n_latents'], t2['k']) == (6, 3, 4)
    assert t2['buffer_size'] == N
    for key in ('generated', 'data_baseline', 'random_baseline',
                'neighbor_state_dist_nearest', 'neighbor_state_dist_kth'):
        assert np.isfinite(t2[key]['mean'])
    assert (t2['neighbor_state_dist_nearest']['mean']
            <= t2['neighbor_state_dist_kth']['mean'])


def test_tool_test_selection_and_untrained_guard(npz, tmp_path):
    from tools.validate_decode_recovery import main

    out = str(tmp_path / 'report1.json')
    main.__wrapped__(_cfg(npz, report_out=out, tests=[1]))
    with open(out) as f:
        report = json.load(f)
    assert 'test1_decode_recovery' in report and 'test2_buffer_recovery' not in report

    with pytest.raises(AssertionError, match='UNTRAINED'):
        main.__wrapped__(_cfg(npz, tests=[1], inversion=dict(batch_size=8,
                                                            allow_untrained=False)))
