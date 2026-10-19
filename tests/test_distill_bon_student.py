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


def _set_student():
    import jax
    from tools.distill_bon_student import make_student
    table = np.random.default_rng(2).standard_normal((3, 6, 4)).astype(np.float32)
    net = make_student(5, 16, 2, ob_dim=3, set_table=table, set_hidden=8)
    x = np.concatenate([np.random.default_rng(3).standard_normal((4, 3)), np.array([[0], [1], [2], [1]])], 1)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 4)))
    # perturb the zero-init head so the output depends on the input
    params = jax.tree_util.tree_map(lambda p: p + 0.1, params)
    return net, params, table, x.astype(np.float32)


def test_set_student_reads_its_task_row():
    from tools.distill_bon_student import make_student
    net, params, table, x = _set_student()
    mu, _ = net.apply(params, jnp.asarray(x))
    assert mu.shape == (4, 5)
    # rows 1 and 3 have the same task; changing only s changes mu
    assert not np.allclose(np.asarray(mu[1]), np.asarray(mu[3]))
    # permuting the goals inside each task's set leaves the output unchanged
    perm = make_student(5, 16, 2, ob_dim=3, set_table=table[:, ::-1].copy(), set_hidden=8)
    np.testing.assert_allclose(np.asarray(perm.apply(params, jnp.asarray(x))[0]), np.asarray(mu), rtol=1e-5, atol=1e-6)


def test_early_stop_keeps_the_best_validation_step():
    from omegaconf import OmegaConf

    from tools.distill_bon_student import train_student
    rng = np.random.default_rng(4)
    x = rng.standard_normal((256, 3)).astype(np.float32)
    v = (x[:, :1] * np.ones((1, 5)) + 0.1 * rng.standard_normal((256, 5))).astype(np.float32)
    cfg = OmegaConf.create({'seed': 0, 'train_steps': 60, 'train_batch': 64, 'hidden': 16, 'n_layers': 1,
                            'early_stop': True, 'val_checks': 6, 'weight_decay': 1e-4})
    import jax
    _, curve, stop, _ = train_student(x, v, x[:32], v[:32], cfg, jax.random.PRNGKey(0))
    best = min(curve, key=lambda r: r['val_nll'])
    assert stop['best_step'] == best['step'] and abs(stop['best_val_nll'] - best['val_nll']) < 1e-6
