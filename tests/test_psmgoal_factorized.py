"""psmgoal measure_form=factorized and the few-policy (max_log_seed=2) arm.

factorized: phi(s,u,g) = A(s,u)^T f(g), b(s,u,g) = beta(s,u)^T f(g), f on the sqrt(f_dim)
sphere, plus f_ortho_coef * ||E[f f^T] - I||_F^2 over the batch's next states. The default
(joint) must stay byte-identical to the code before the flag: the golden digest below was
computed from agents/psmgoal.py as it stood before measure_form was added (2026-10-01, CPU,
this file's own `_digest`).

Run in its own process: JAX_PLATFORMS=cpu .venv/bin/python -m pytest
tests/test_psmgoal_factorized.py -q -p no:cacheprovider
"""
import hashlib
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from agents.psmgoal import PSMGoalAgent, f_ortho_loss, get_config
from utils.psm_networks import FactorizedMeasure, RLUMeasure
from utils.psm_proto import proto_seed_ints, sample_z_bin

OB, DA, Z, CODE, N, FD = 6, 3, 8, 4, 8, 5

# sha256 of every field's leaves + every info value after 3 default updates, pre-flag code.
GOLDEN_JOINT = "f9156145f0046205f84b1fd181523a25e870bd12f521164bcc1a78398e642f99"


def _cfg(**over):
    c = get_config()
    c.z_dim = Z
    c.max_log_seed = CODE
    c.batch_size = N
    c.k_goals = 4
    c.gpi_num_u = 5
    c.num_inference_steps = 2
    c.infer_batch = N
    c.allow_untrained_flow = True
    c.measure.hidden_dim = 16
    c.w.hidden_dim = 16
    c.l.hidden_dim = 16
    c.actor.hidden_dim = 16
    c.flow.hidden_dims = (16, 16)
    c.f_dim = FD
    for k, v in over.items():
        setattr(c, k, v)
    return c


def _agent(**over):
    return PSMGoalAgent.create(0, np.zeros((2, OB), np.float32), np.zeros((2, DA), np.float32),
                               _cfg(**over))


def _batch(n=N, seed=0):
    rng = np.random.default_rng(seed)
    r = -np.ones((n,), np.float32)
    r[:2] = 0.0
    return {
        "observations": rng.standard_normal((n, OB)).astype(np.float32),
        "next_observations": rng.standard_normal((n, OB)).astype(np.float32),
        "noise_preimage": rng.standard_normal((n, DA)).astype(np.float32),
        "index": np.arange(n).astype(np.int32),
        "rewards": r,
    }


_FIELDS = ("rng", "basis", "w", "l", "actor", "w_star", "target_basis", "target_w",
           "eval_goals", "eval_w_star")


def _digest(**over):
    ag = PSMGoalAgent.create(0, np.zeros((2, OB), np.float32), np.zeros((2, DA), np.float32),
                             _cfg(**over))
    h = hashlib.sha256()
    for step in range(3):
        ag, info = ag.update(_batch(seed=step))
        for k in sorted(info):
            h.update(k.encode())
            h.update(np.asarray(info[k], np.float32).tobytes())
    for name in _FIELDS:
        for leaf in jax.tree_util.tree_leaves(getattr(ag, name)):
            h.update(np.asarray(leaf).tobytes())
    return h.hexdigest()


def test_defaults():
    c = get_config()
    assert c.measure_form == "joint" and c.f_dim == 128 and c.f_ortho_coef == 1.0


def test_joint_byte_identical():
    # f_dim is set in _cfg but is read only when factorized.
    assert _digest() == GOLDEN_JOINT


def test_basis_class_per_form():
    assert isinstance(_agent().basis.model_def, RLUMeasure)
    ag = _agent(measure_form="factorized")
    assert isinstance(ag.basis.model_def, FactorizedMeasure)
    assert ag.basis.model_def.f_dim == FD


def test_factorized_shapes_and_hand_computed_M():
    ag = _agent(measure_form="factorized")
    b = _batch()
    obs, u, g = (jnp.asarray(b[k]) for k in ("observations", "noise_preimage", "next_observations"))
    A, beta = ag.basis(obs, u, method="operators")
    f = ag.basis(g, method="features")
    assert A.shape == (N, FD, Z) and beta.shape == (N, FD) and f.shape == (N, FD)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(f), axis=-1), np.sqrt(FD), rtol=1e-5)
    phi, bb = ag.mesh_phi_b(obs, u, g)
    assert phi.shape == (N, N, Z) and bb.shape == (N, N)
    w = ag.w(sample_z_bin(jax.random.PRNGKey(1), N, CODE))
    M = np.asarray(ag.mesh_M(obs, u, g, w))
    An, bn, fn, wn = (np.asarray(x, np.float64) for x in (A, beta, f, w))
    M_hand = np.einsum('ikd,jk,id->ij', An, fn, wn) + np.einsum('ik,jk->ij', bn, fn)
    np.testing.assert_allclose(M, M_hand, rtol=1e-4, atol=1e-4)
    # pointwise __call__ agrees with the mesh
    p1, b1 = ag.basis(obs[1:2], u[1:2], g[3:4])
    np.testing.assert_allclose(np.asarray(p1[0]), np.asarray(phi[1, 3]), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(float(b1[0]), float(bb[1, 3]), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(float(ag.M(obs[1:2], u[1:2], g[3:4], w[1])[0]), M[1, 3],
                               rtol=1e-4, atol=1e-4)


def test_f_ortho_loss():
    # orthonormal-in-expectation pool: rows sqrt(K) e_k cycled -> E[f f^T] = I -> loss 0
    K = 4
    f = np.sqrt(K) * np.eye(K, dtype=np.float32)
    assert abs(float(f_ortho_loss(jnp.asarray(np.tile(f, (3, 1)))))) < 1e-6
    # rank-one pool on the sqrt(K) sphere: E[f f^T] = K e1 e1^T -> (K-1)^2 + (K-1)
    f1 = np.zeros((6, K), np.float32)
    f1[:, 0] = np.sqrt(K)
    np.testing.assert_allclose(float(f_ortho_loss(jnp.asarray(f1))), (K - 1) ** 2 + (K - 1), rtol=1e-5)


def test_f_ortho_term_enters_loss_only_when_factorized():
    b = _batch()
    z = sample_z_bin(jax.random.PRNGKey(0), N, CODE)
    for form in ("joint", "factorized"):
        ag0 = _agent(measure_form=form, f_ortho_coef=0.0)
        ag1 = _agent(measure_form=form, f_ortho_coef=1.0)
        u_next = ag0.proto_bootstrap(z, jnp.arange(N))
        l0, _ = ag0.measure_loss(ag0.basis.params, ag0.w.params, b, z, u_next)
        l1, i1 = ag1.measure_loss(ag1.basis.params, ag1.w.params, b, z, u_next)
        if form == "joint":
            assert float(l0) == float(l1) and "f_ortho_loss" not in i1
        else:
            np.testing.assert_allclose(float(l1) - float(l0), float(i1["f_ortho_loss"]), rtol=1e-4)
            # the f_ortho gradient reaches f only through the f_net params
            basis1, s_plus = ag1.basis, jnp.asarray(b["next_observations"])
            g = jax.grad(lambda p, bs=basis1, sp=s_plus: f_ortho_loss(bs(sp, params=p, method="features")))(basis1.params)
            assert float(sum(jnp.abs(x).sum() for x in jax.tree_util.tree_leaves(g["f_net"]))) > 0
            assert all(float(jnp.abs(x).sum()) == 0 for k, v in g.items() if k != "f_net"
                       for x in jax.tree_util.tree_leaves(v))


def test_f_ortho_descent_reduces_term():
    ag = _agent(measure_form="factorized")
    g = jnp.asarray(np.random.default_rng(3).standard_normal((64, OB)).astype(np.float32))
    loss = lambda p: f_ortho_loss(ag.basis(g, params=p, method="features"))
    p = ag.basis.params
    l_start = float(loss(p))
    for _ in range(50):
        gr = jax.grad(loss)(p)
        p = jax.tree_util.tree_map(lambda a, b: a - 0.01 * b, p, gr)
    assert float(loss(p)) < l_start


def test_one_finite_update_per_form():
    for form in ("joint", "factorized"):
        ag = _agent(measure_form=form)
        new, info = ag.update(_batch())
        for k, v in info.items():
            assert np.all(np.isfinite(np.asarray(v))), (form, k)
        moved = jax.tree_util.tree_map(lambda a, b: float(jnp.abs(a - b).max()),
                                       new.basis.params, ag.basis.params)
        assert max(jax.tree_util.tree_leaves(moved)) > 0
        if form == "factorized":
            assert "f_ortho_loss" in info


def test_factorized_inference_paths_run():
    for coef in ("lp", "regression"):
        ag = _agent(measure_form="factorized", coef_source=coef)
        ag = ag.infer_eval_goals(_batch(), _batch()["rewards"] + 1.0)
        np.testing.assert_allclose(float(jnp.linalg.norm(ag.eval_w_star)), np.sqrt(Z), rtol=1e-4)
        a = ag.sample_actions(jnp.zeros((OB,)), seed=jax.random.PRNGKey(0))
        assert a.shape == (DA,) and np.all(np.isfinite(np.asarray(a)))


def test_max_log_seed_2_samples_four_codes():
    z = np.asarray(sample_z_bin(jax.random.PRNGKey(0), 4096, 2))
    assert z.shape == (4096, 2)
    codes = {tuple(r) for r in z}
    assert codes == {(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)}
    seeds = np.asarray(proto_seed_ints(jnp.asarray(z), np.zeros(4096, np.int32), 2))
    assert set(seeds.tolist()) == {0, 1, 2, 3}
    ag = _agent(max_log_seed=2)
    w = np.asarray(ag.w(jnp.asarray(z)))
    assert w.shape == (4096, Z)
    assert len({tuple(np.round(r, 5)) for r in w}) == 4
    _, info = ag.update(_batch())
    assert all(np.all(np.isfinite(np.asarray(v))) for v in info.values())
