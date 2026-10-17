"""tools/distill_bon_student.py pure helpers: the Gaussian student's density and the episode split.

Run: JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_distill_bon_student.py -q
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
from scipy.stats import multivariate_normal

from tools.distill_bon_student import gaussian_nll, split_episodes, task_env_name, tril_from_raw


def test_task_env_name():
    assert task_env_name("cube-single-play-singletask-v0", 3) == "cube-single-play-singletask-task3-v0"
    assert task_env_name("cube-single-play-singletask-task1-v0", 5) == "cube-single-play-singletask-task5-v0"


def test_tril_is_lower_with_positive_diagonal():
    d = 5
    raw = jnp.asarray(np.random.default_rng(0).standard_normal((7, d * (d + 1) // 2)) * 3, jnp.float32)
    L = np.asarray(tril_from_raw(raw, d))
    assert L.shape == (7, d, d)
    assert np.allclose(np.triu(L, 1), 0)
    assert (np.diagonal(L, axis1=1, axis2=2) > 0).all()


def test_nll_matches_scipy():
    d, n = 5, 6
    rng = np.random.default_rng(1)
    raw = jnp.asarray(rng.standard_normal((n, d * (d + 1) // 2)), jnp.float32)
    L = tril_from_raw(raw, d)
    mu = jnp.asarray(rng.standard_normal((n, d)), jnp.float32)
    v = jnp.asarray(rng.standard_normal((n, d)), jnp.float32)
    got = np.asarray(gaussian_nll(v, mu, L))
    Ln = np.asarray(L, np.float64)
    want = [-multivariate_normal(np.asarray(mu[i]), Ln[i] @ Ln[i].T).logpdf(np.asarray(v[i])) for i in range(n)]
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_zero_init_is_near_prior():
    """raw = 0 gives L = (softplus(0) + 1e-3) I, mu = 0: a scaled standard normal."""
    d = 5
    L = np.asarray(tril_from_raw(jnp.zeros((1, d * (d + 1) // 2)), d))[0]
    np.testing.assert_allclose(L, (np.log(2) + 1e-3) * np.eye(d), rtol=1e-5)


def test_split_holds_out_whole_episodes():
    ep = np.repeat(np.arange(20), 7)
    tr = split_episodes(ep, 0.1, 0)
    assert (~tr).sum() == 2 * 7
    for e in np.unique(ep):
        assert len(set(tr[ep == e])) == 1
    assert split_episodes(ep, 0.0, 0).all()
