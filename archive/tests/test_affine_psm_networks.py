import jax, jax.numpy as jnp, numpy as np
from utils.psm_networks import AffineMeasureNet, LagrangeNet


def _io(B=4, obs=8, act=2, d=5):
    rng = np.random.default_rng(0)
    o = jnp.asarray(rng.standard_normal((B, obs)), jnp.float32)
    a = jnp.asarray(np.clip(rng.standard_normal((B, act)), -1, 1), jnp.float32)
    x = jnp.asarray(rng.standard_normal((B, obs)), jnp.float32)
    return o, a, x, d


def test_affine_measure_shapes():
    o, a, x, d = _io()
    net = AffineMeasureNet(d_dim=d, hidden_dim=16, hidden_layers=2)
    params = net.init(jax.random.PRNGKey(0), o, a, x)["params"]
    phi, b = net.apply({"params": params}, o, a, x)
    assert phi.shape == (o.shape[0], d)
    assert b.shape == (o.shape[0], 1)


def test_measure_is_affine_in_w():
    # M(w) = phi·w + b must be exactly affine: M(w1)-M(w2) == phi·(w1-w2).
    o, a, x, d = _io()
    net = AffineMeasureNet(d_dim=d, hidden_dim=16, hidden_layers=2)
    params = net.init(jax.random.PRNGKey(0), o, a, x)["params"]
    phi, b = net.apply({"params": params}, o, a, x)
    w1 = jnp.asarray(np.random.default_rng(1).standard_normal((d,)), jnp.float32)
    w2 = jnp.asarray(np.random.default_rng(2).standard_normal((d,)), jnp.float32)
    M1 = (phi * w1).sum(-1, keepdims=True) + b
    M2 = (phi * w2).sum(-1, keepdims=True) + b
    lhs = M1 - M2
    rhs = (phi * (w1 - w2)).sum(-1, keepdims=True)
    assert np.allclose(np.asarray(lhs), np.asarray(rhs), atol=1e-5)


def test_lagrange_nonnegative():
    o, a, x, _ = _io()
    net = LagrangeNet(hidden_dim=16, hidden_layers=2)
    params = net.init(jax.random.PRNGKey(0), o, a, x)["params"]
    lam = net.apply({"params": params}, o, a, x)
    assert lam.shape == (o.shape[0], 1)
    assert np.all(np.asarray(lam) >= 0.0)
