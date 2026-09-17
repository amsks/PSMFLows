"""dataset.reward_override_path and tools/relabel_reward_rhat.py.

The override replaces ONE array, row for row, and refuses anything else: a wrong-length
file (the wrong env or a `--limit` plumbing file) must fail at load, never train. The
tool's w is the eval's w (`infer_eval_z` on the shifted reward), and r_hat is phi^T w with
no rescale, so the relabel is the reward channel the zero-shot agent deploys.
"""
import numpy as np
import pytest

from tests.test_psmflow_agent import ACT, OBS, _agent, _batch
from tools.relabel_reward_rhat import infer_w, relabel_stats, rhat_in_batches, scale_to_real
from utils.datasets import apply_reward_override


def _toy(n=12):
    rng = np.random.default_rng(0)
    terminals = np.zeros(n, np.float32)
    terminals[-1] = 1.0
    return {
        'observations': rng.standard_normal((n, OBS)).astype(np.float32),
        'actions': rng.standard_normal((n, ACT)).astype(np.float32),
        'next_observations': rng.standard_normal((n, OBS)).astype(np.float32),
        'rewards': np.full(n, -1.0, np.float32),
        'masks': np.ones(n, np.float32),
        'terminals': terminals,
    }


def _write(tmp_path, rewards, name='r.npz'):
    path = tmp_path / name
    np.savez(path, rewards=np.asarray(rewards))
    return str(path)


def test_override_replaces_rewards_and_nothing_else(tmp_path):
    d = _toy()
    new = np.linspace(-2.0, 3.0, len(d['rewards']))
    out = apply_reward_override(d, _write(tmp_path, new.astype(np.float64)))
    assert out['rewards'].dtype == np.float32
    assert np.allclose(out['rewards'], new)
    for k in d:
        if k != 'rewards':
            assert np.array_equal(out[k], d[k]), k
    assert np.all(d['rewards'] == -1.0), 'the input dict must not be mutated'


def test_override_refuses_a_length_mismatch(tmp_path):
    d = _toy()
    with pytest.raises(AssertionError, match='rows'):
        apply_reward_override(d, _write(tmp_path, np.zeros(len(d['rewards']) - 1)))


def test_override_refuses_non_finite(tmp_path):
    d = _toy()
    r = np.zeros(len(d['rewards']))
    r[3] = np.nan
    with pytest.raises(AssertionError, match='non-finite'):
        apply_reward_override(d, _write(tmp_path, r))


def test_relabel_stats_on_a_perfect_and_a_random_readout():
    rng = np.random.default_rng(0)
    r = np.full(1000, -1.0)
    r[rng.choice(1000, 20, replace=False)] = 0.0
    s = relabel_stats(r + 1.0, r, reward_shift=1.0, top_frac=0.02)
    assert s['corr'] == pytest.approx(1.0, abs=1e-6)
    assert s['r2_raw'] == pytest.approx(1.0, abs=1e-6)
    assert s['r2_affine'] == pytest.approx(1.0, abs=1e-6)
    assert s['top_k'] == 20 and s['top_precision'] == 1.0
    assert s['base_rate'] == pytest.approx(0.02)
    # A scaled-and-shifted copy: raw R^2 drops, affine R^2 does not.
    s2 = relabel_stats(3.0 * (r + 1.0) + 5.0, r, reward_shift=1.0)
    assert s2['r2_raw'] < 0.0 and s2['r2_affine'] == pytest.approx(1.0, abs=1e-6)
    assert s2['affine_scale'] == pytest.approx(1.0 / 3.0, abs=1e-6)
    noise = relabel_stats(rng.standard_normal(1000), r, reward_shift=1.0)
    assert abs(noise['corr']) < 0.2 and noise['r2_affine'] < 0.05


def test_tool_w_matches_infer_eval_z_and_rhat_is_phi_dot_w():
    agent = _agent()
    b = _batch()
    n = b['observations'].shape[0]
    rewards = np.where(np.arange(n) % 7 == 0, 0.0, -1.0).astype(np.float32)
    w = infer_w(agent, b['next_observations'], rewards, reward_shift=1.0)
    ref = np.asarray(agent.infer_eval_z(b['next_observations'], rewards + 1.0).task_z)
    assert w.shape == (agent.config['z_dim'],)
    assert np.allclose(w, ref, atol=1e-6)
    assert np.linalg.norm(w) == pytest.approx(np.sqrt(agent.config['z_dim']), rel=1e-4)
    r_hat = rhat_in_batches(agent, b['next_observations'], w, batch_size=5)
    phi = np.asarray(agent.phi(b['next_observations']))
    assert r_hat.shape == (b['next_observations'].shape[0],)
    assert np.allclose(r_hat, phi @ w, atol=1e-5)


def test_scale_to_real_recovers_the_dataset_convention():
    rng = np.random.default_rng(1)
    r = np.full(2000, -1.0, np.float32)
    r[rng.choice(2000, 40, replace=False)] = 0.0
    r_hat = 3.0 * (r + 1.0) + 5.0                     # a scaled, offset copy of the shifted reward
    out, scale, offset = scale_to_real(r_hat, r, reward_shift=1.0)
    assert out.dtype == np.float32
    assert scale == pytest.approx(1.0 / 3.0, abs=1e-6) and offset == pytest.approx(-5.0 / 3.0, abs=1e-5)
    assert np.allclose(out, r, atol=1e-4)                # back on the -1/0 convention
    assert relabel_stats(out, r, reward_shift=0.0)['corr'] == pytest.approx(1.0, abs=1e-6)
