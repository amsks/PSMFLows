"""Factorized affine measure: Phi(s,a,x) = A(s,a) phi_x(x)  (write-up Prop. 4.3 under A5).

The point of the factorization is cost — it turns the B^2 contrastive mesh into B network
evaluations plus two matmuls, which is what makes batch_size=1024 (the reference cube
value) affordable. These tests pin the two things that must survive that change: the mesh
must equal the naive elementwise computation, and M must stay LINEAR in w (the property
the constrained-LP `full` inference depends on).
"""
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf

from agents import agents
from utils.psm_networks import FactoredAffineMeasureNet


def _io(B=6, obs=8, act=2, d=5, k=4):
    rng = np.random.default_rng(0)
    o = jnp.asarray(rng.standard_normal((B, obs)), jnp.float32)
    a = jnp.asarray(np.clip(rng.standard_normal((B, act)), -1, 1), jnp.float32)
    x = jnp.asarray(rng.standard_normal((B, obs)), jnp.float32)
    net = FactoredAffineMeasureNet(d_dim=d, k_dim=k, hidden_dim=16, hidden_layers=2)
    params = net.init(jax.random.PRNGKey(0), o, a, x)["params"]
    return net, params, o, a, x, d


def test_shapes():
    net, params, o, a, x, d = _io()
    phi, b = net.apply({"params": params}, o, a, x)
    assert phi.shape == (o.shape[0], d)
    assert b.shape == (o.shape[0], 1)


def test_x_basis_is_sqrt_k_normalized():
    # phi_x plays the reference PhiMap's role, so it carries the sqrt(k) normalization.
    # This is what keeps the ortho regularizer's diagonal term inert at ortho_coef=1000.
    net, params, o, a, x, _ = _io()
    px, _ = net.apply({"params": params}, x, method="x")   # (phi_x, phi_b)
    k = px.shape[-1]
    assert np.allclose(np.linalg.norm(np.asarray(px), axis=-1), np.sqrt(k), atol=1e-4)


def test_measure_is_affine_in_w():
    # M(w) = Phi·w + b must be exactly affine: M(w1) - M(w2) == Phi·(w1 - w2).
    # The constrained-LP inference is a genuine program in w only if this holds.
    net, params, o, a, x, d = _io()
    phi, b = net.apply({"params": params}, o, a, x)
    rng = np.random.default_rng(1)
    w1 = jnp.asarray(rng.standard_normal((d,)), jnp.float32)
    w2 = jnp.asarray(rng.standard_normal((d,)), jnp.float32)
    m1 = (phi * w1).sum(-1) + b.squeeze(-1)
    m2 = (phi * w2).sum(-1) + b.squeeze(-1)
    assert np.allclose(np.asarray(m1 - m2), np.asarray((phi * (w1 - w2)).sum(-1)), atol=1e-5)


def test_mesh_equals_naive_elementwise():
    # THE load-bearing test: the cheap (B,B) mesh must reproduce, entry for entry, the
    # naive computation that evaluates the net on every (s_i, a_i, x_j) triple.
    net, params, o, a, x, d = _io(B=6)
    B = o.shape[0]
    rng = np.random.default_rng(2)
    w = jnp.asarray(rng.standard_normal((d,)), jnp.float32)
    b_scale = net.b_scale

    # cheap path (what agents/affine_psm.py:measure_loss does)
    A, beta, px, pb = net.apply({"params": params}, o, a, x, method="mesh_terms")
    psi = jnp.einsum("bdk,d->bk", A, w)
    mesh = psi @ px.T + b_scale * jnp.tanh(beta @ pb.T)

    # naive path: evaluate the module on all B^2 triples
    i_idx = jnp.repeat(jnp.arange(B), B)
    j_idx = jnp.tile(jnp.arange(B), B)
    phi_n, b_n = net.apply({"params": params}, o[i_idx], a[i_idx], x[j_idx])
    naive = ((phi_n * w).sum(-1, keepdims=True) + b_n).reshape(B, B)

    assert np.allclose(np.asarray(mesh), np.asarray(naive), atol=1e-4), \
        np.abs(np.asarray(mesh) - np.asarray(naive)).max()


def _config(factored=True):
    with initialize(version_base="1.3", config_path="../configs/agent"):
        cfg = compose(config_name="affine_psm")
    config = ml_collections.ConfigDict(OmegaConf.to_container(cfg, resolve=True))
    config["measure"]["factored"] = factored
    config["measure"]["hidden_dim"] = 32
    config["measure"]["hidden_layers"] = 2
    config["actor"]["flow_actor_hidden_dim"] = 32
    config["actor"]["flow_vf_hidden_dim"] = 32
    config["actor"]["flow_vf_hidden_layers"] = 2
    config["batch_size"] = 8
    return config


def _batch(n=8, obs=8, act=2):
    rng = np.random.default_rng(0)
    return dict(
        observations=rng.standard_normal((n, obs)).astype(np.float32),
        actions=np.clip(rng.standard_normal((n, act)), -1, 1).astype(np.float32),
        next_observations=rng.standard_normal((n, obs)).astype(np.float32),
        index=np.arange(n, dtype=np.int64),
        masks=np.ones((n,), np.float32),
    )


def test_factored_agent_trains():
    config = _config(factored=True)
    agent = agents["affine_psm"].create(0, np.zeros((1, 8), np.float32),
                                        np.zeros((1, 2), np.float32), config)
    assert agent.config["measure_factored"] is True
    b = _batch()
    losses = []
    for _ in range(60):
        agent, info = agent.update(b)
        losses.append(float(info["psm_offdiag"]))
    assert np.isfinite(losses[-1])
    assert np.mean(losses[-10:]) < np.mean(losses[:10]), (losses[:3], losses[-3:])


def test_unfactored_path_still_works():
    config = _config(factored=False)
    agent = agents["affine_psm"].create(0, np.zeros((1, 8), np.float32),
                                        np.zeros((1, 2), np.float32), config)
    assert agent.config["measure_factored"] is False
    agent, info = agent.update(_batch())
    assert np.isfinite(float(info["psm_offdiag"]))


def test_codebook_draws_B_distinct_codes():
    """RLU draws B codes per batch (sample_z(batch_size), psm.py:431), not one.

    We previously broadcast a single code across the batch, training the basis on ONE
    proto-policy per step instead of B. The codes must also be row-aligned: row i's
    bootstrap action is pi_{z_i}(s'_i), the alignment discrete_psm.py:596 gets right.
    """
    import jax
    config = _config(factored=True)
    agent = agents["affine_psm"].create(0, np.zeros((1, 8), np.float32),
                                        np.zeros((1, 2), np.float32), config)
    sampled = agent.sample_step_inputs(_batch(), jax.random.PRNGKey(0))
    z = np.asarray(sampled.proto_seed)
    assert z.shape == (8, config["max_log_seed"])
    assert len(np.unique(z, axis=0)) > 1, "all rows share one codebook code"
    # distinct codes must give distinct bootstrap actions
    na = np.asarray(sampled.proto_next_action)
    assert len(np.unique(na, axis=0)) > 1


def test_full_mode_redistills_the_actor():
    """`full` inference must leave the actor conditioned on the w_inf it solved for.

    The amortized actor is trained on the closed-form w_g only; d80f7f0 dropped eval-time
    distillation, so `full` was acting off-distribution. infer_eval(full) must move the
    actor; infer_eval(zero_shot) must not (its w IS the training-time w).
    """
    import jax.numpy as jnp

    class _DS:
        def __init__(self, n=64, obs=8, act=2):
            r = np.random.default_rng(0)
            self.obs = r.standard_normal((n, obs)).astype(np.float32)
            self.act = np.clip(r.standard_normal((n, act)), -1, 1).astype(np.float32)
            self.nxt = r.standard_normal((n, obs)).astype(np.float32)
            self.n = n
            self._r = np.random.default_rng(1)

        def sample(self, k):
            i = self._r.integers(0, self.n, size=k)
            return dict(observations=self.obs[i], actions=self.act[i],
                        next_observations=self.nxt[i], index=i.astype(np.int64))

    def _leaves(t):
        return [np.asarray(x) for x in jax.tree_util.tree_leaves(t)]

    config = _config(factored=True)
    config["inference"]["num_inference_steps"] = 5
    config["inference"]["num_actor_inference_steps"] = 5
    ds = _DS()
    goal = ds.sample(1)["next_observations"][0]

    config["inference"]["mode"] = "full"
    agent = agents["affine_psm"].create(0, np.zeros((1, 8), np.float32),
                                        np.zeros((1, 2), np.float32), config)
    before = _leaves(agent.actor.params)
    tuned = agent.infer_eval(ds, goal)
    moved = max(np.abs(x - y).max() for x, y in zip(before, _leaves(tuned.actor.params)))
    assert moved > 0, "full mode left the actor conditioned on the training-time w"

    config["inference"]["mode"] = "zero_shot"
    agent2 = agents["affine_psm"].create(0, np.zeros((1, 8), np.float32),
                                         np.zeros((1, 2), np.float32), config)
    b2 = _leaves(agent2.actor.params)
    zs = agent2.infer_eval(ds, goal)
    assert all(np.allclose(x, y) for x, y in zip(b2, _leaves(zs.actor.params)))
