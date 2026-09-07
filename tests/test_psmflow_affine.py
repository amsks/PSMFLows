"""Affine psi head: psi(s, u, u') = A(s,u)^T w(u') + beta(s,u) (write-up Prop. bilinear).

Rem. `tradeoff` records that the ORIGINAL agent adopted the bilinear form but NOT the
affineness of psi in the policy coordinate -- w^{u'} was absorbed into a free network.
`psi_form=affine` restores it explicitly, with w(.) a learned encoder off the policy latent
(the paper asserts w^{u'} exists and gives no formula; the encoder is our design choice,
see docs/design/2026-09-04-affine-psi.md). Since 2026-09-04 it is THE DEFAULT: cube 0.532 /
0.620 at 250k against 0.083 for `psi_form=free`.

What is pinned here:
  - `_agent()` with no flags IS this head -- byte-identical to the explicit affine arm;
  - `psi_form=free` is still reachable and still builds a structurally different psi;
  - psi is EXACTLY affine in w(u'): psi(u'_1) - psi(u'_2) = A^T (w(u'_1) - w(u'_2));
  - A and beta do not see the policy index (Assumption `affine`), so a psi difference
    across u' at fixed (s,u) lives in the column space of A alone;
  - the head takes a finite step in both shipped arms (strict GPI and latent actor);
  - psi_form=affine with policy_index=task_vector is refused, not silently mis-typed;
  - acting works in both modes;
  - the collapse diagnostics (w_enc_spread, psi_q_*_rel) are emitted in-loop.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_psmflow_agent import ACT, OBS, _agent, _batch


def _affine(**overrides):
    kw = {"psi_form": "affine", "policy_index": "latent",
          "train_actor": False, "acting": "gpi"}
    kw.update(overrides)
    return _agent(**kw)


def _leaves(agent):
    return jax.tree_util.tree_leaves((agent.phi.params, agent.psi.params,
                                      agent.actor.params, agent.actor_vf.params))


def test_the_default_agent_is_the_affine_head():
    """No flags => this head, byte for byte: an update of `_agent()` and an update of the
    explicit affine arm leave identical params and identical losses."""
    a = _agent()
    assert a.config["psi_form"] == "affine"
    assert a.config["policy_index"] == "latent"
    b = _affine()
    a2, ai = a.update(_batch())
    b2, bi = b.update(_batch())
    for x, y in zip(_leaves(a2), _leaves(b2)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    for k in ("psm_loss", "orth_loss"):
        assert float(ai[k]) == float(bi[k])
    # The affine/latent diagnostics ride along on the default path now.
    assert "psi_q_spread_rel" in ai and "w_enc_spread" in ai


def test_free_head_is_still_reachable_and_differs():
    """psi_form=free is the ablation: same call signature, structurally different psi and a
    different update. (It is the arm that measured 0.083 where affine measures 0.532.)"""
    free = _agent(psi_form="free")
    assert free.config["psi_form"] == "free"
    free2, fi = free.update(_batch())
    assert math.isfinite(float(fi["psm_loss"]))
    aff_leaves = jax.tree_util.tree_leaves(_affine().psi.params)
    free_leaves = jax.tree_util.tree_leaves(free.psi.params)
    assert len(aff_leaves) != len(free_leaves), "the two heads must not be the same network"
    # The free head has no w(.) encoder, so it emits no collapse diagnostic.
    assert "w_enc_spread" not in fi


def test_psi_is_exactly_affine_in_the_policy_coordinate():
    """psi(s,u,u'_1) - psi(s,u,u'_2) = A(s,u)^T (w(u'_1) - w(u'_2)), numerically."""
    agent = _affine()
    rng = np.random.default_rng(0)
    n = 8
    obs = rng.standard_normal((n, OBS)).astype(np.float32)
    u = rng.standard_normal((n, ACT)).astype(np.float32)
    i1 = rng.standard_normal((n, ACT)).astype(np.float32)
    i2 = rng.standard_normal((n, ACT)).astype(np.float32)

    p1 = np.asarray(agent.psi(obs, i1, u))                  # (P, n, z)
    p2 = np.asarray(agent.psi(obs, i2, u))
    w1 = np.asarray(agent.psi(i1, method="encode_index"))   # (n, d_w)
    w2 = np.asarray(agent.psi(i2, method="encode_index"))
    A, beta = agent.psi(obs, u, method="sa_terms")
    A, beta = np.asarray(A), np.asarray(beta)               # (P, n, z, d_w), (P, n, z)

    pred = np.einsum("pnzw,nw->pnz", A, w1 - w2)
    np.testing.assert_allclose(p1 - p2, pred, rtol=1e-4, atol=1e-4)
    # The value itself is A^T w + beta -- beta is the only u'-independent part.
    np.testing.assert_allclose(p1, np.einsum("pnzw,nw->pnz", A, w1) + beta,
                               rtol=1e-4, atol=1e-4)


def test_A_and_beta_do_not_depend_on_the_policy_index():
    """Assumption `affine`: the basis and the bias are policy-index free."""
    agent = _affine()
    rng = np.random.default_rng(1)
    obs = rng.standard_normal((4, OBS)).astype(np.float32)
    u = rng.standard_normal((4, ACT)).astype(np.float32)
    A1, b1 = agent.psi(obs, u, method="sa_terms")
    A2, _b2 = agent.psi(obs, u, method="sa_terms")
    np.testing.assert_array_equal(np.asarray(A1), np.asarray(A2))
    # sa_terms takes no index argument at all -- that IS the guarantee. Sanity: A is wide.
    assert np.asarray(A1).shape[-1] == agent.config["affine"]["w_dim"]
    assert np.asarray(b1).shape[-1] == agent.config["z_dim"]


def test_w_encoder_is_on_the_unit_sphere_and_uses_its_input():
    agent = _affine()
    idx = np.random.default_rng(2).standard_normal((32, ACT)).astype(np.float32)
    w = np.asarray(agent.psi(idx, method="encode_index"))
    np.testing.assert_allclose(np.linalg.norm(w, axis=-1), 1.0, rtol=1e-5)
    assert w.std(0).mean() > 1e-3, "the encoder ignores u' -- psi would be index-free"


def test_strict_arm_takes_a_finite_step():
    """policy_index=latent, train_actor=false, acting=gpi -- the paper's agent."""
    agent = _affine()
    before = _leaves(agent)
    agent2, info = agent.update(_batch())
    for k in ("psm_loss", "orth_loss", "psi_q_spread_rel", "psi_q_index_spread_rel",
              "w_enc_spread"):
        assert k in info, k
        assert math.isfinite(float(info[k])), (k, info[k])
    assert "actor_loss" not in info
    assert not all(bool(jnp.array_equal(x, y)) for x, y in zip(before, _leaves(agent2)))
    assert float(info["w_enc_spread"]) > 1e-3, "encoder collapsed at init"


def test_actor_arm_takes_a_finite_step():
    """policy_index=latent, train_actor=true, acting=actor -- the DSRL-style arm."""
    agent = _affine(train_actor=True, acting="actor")
    agent2, info = agent.update(_batch())
    for k in ("psm_loss", "actor_loss", "actor_q", "actor_bc_error", "w_enc_spread"):
        assert math.isfinite(float(info[k])), (k, info[k])
    assert not all(bool(jnp.array_equal(x, y)) for x, y in
                   zip(jax.tree_util.tree_leaves(agent.actor.params),
                       jax.tree_util.tree_leaves(agent2.actor.params))), "actor did not train"


def test_affine_with_a_task_vector_index_is_refused():
    with pytest.raises(AssertionError, match="psi_form=affine requires policy_index=latent"):
        _agent(psi_form="affine", policy_index="task_vector")


def test_unknown_psi_form_is_refused():
    with pytest.raises(AssertionError, match="psi_form"):
        _agent(psi_form="bilinear", policy_index="latent")


@pytest.mark.parametrize("acting,train_actor", [("gpi", False), ("actor", True)])
def test_sample_actions_runs(acting, train_actor):
    agent = _affine(acting=acting, train_actor=train_actor, gpi_num_u=8)
    a = agent.sample_actions(_batch()["observations"][0], seed=jax.random.PRNGKey(0))
    assert a.shape == (ACT,)
    assert np.all(np.abs(np.asarray(a)) <= 1.0 + 1e-6)


def test_psi_params_carry_the_encoder_and_the_wide_A_head():
    """A checkpoint of this head is structurally different from the free one; restoring
    one into the other must not half-succeed (tools/eval_checkpoint.py inherits psi_form
    from the run's flags.json, which is what keeps them apart)."""
    aff = jax.tree_util.tree_leaves(_affine().psi.params)
    free = jax.tree_util.tree_leaves(_agent(psi_form="free", policy_index="latent",
                                            train_actor=False, acting="gpi").psi.params)
    assert sum(x.size for x in aff) != sum(x.size for x in free)
    w_dim, z = 128, _affine().config["z_dim"]
    assert any(x.shape[-1] == z * w_dim for x in aff if x.ndim >= 1), "no wide A head"
