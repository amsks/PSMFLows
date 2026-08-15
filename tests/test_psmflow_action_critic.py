"""Idea-1 action branch: psi_a(s, w, a) + eps-bounded residual (2026-08-13).

Pins:
  - disabled (default): psi/phi/actor/actor_vf one-step deltas byte-identical to an
    enabled run's shared branches — the new branch must not touch the shared updates,
    the rng stream, or the sampled inputs;
  - disabled: psi_a and residual params never move;
  - enabled: psi_a and residual train with finite losses; residual_spend <= eps;
  - enabled acting: action valid and within eps (per-dim) of the pure decode.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

from tests.test_psmflow_agent import _agent, _batch


def _tree_equal(a, b):
    return all(bool(jnp.array_equal(x, y))
               for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


# One place for the branch config the tests build agents from, so a new knob in
# configs/agent/psmflow.yaml does not silently diverge from what is exercised here.
_AC = dict(enabled=True, discount=0.99, tau=0.005, pessimism=0.0, lr=3.0e-4,
           hidden_dim=512, hidden_layers=1, embedding_layers=2, spread_candidates=8,
           eval_rank_k=0, fb_graft=False, ortho_coef=1000.0, b_tau=0.005)


def _enabled_agent(**kw):
    return _agent(action_critic=dict(_AC), **kw)


def _pessimistic_agent(lam, **kw):
    return _agent(action_critic=dict(_AC, pessimism=lam), **kw)


def test_disabled_is_inert_and_shared_branches_match_enabled():
    off = _agent()
    on = _enabled_agent()
    off2, _ = off.update(_batch())
    on2, info = on.update(_batch())
    # Shared branches identical across the switch.
    assert _tree_equal(off2.psi.params, on2.psi.params), "enabling changed the psi step"
    assert _tree_equal(off2.phi.params, on2.phi.params), "enabling changed the phi step"
    assert _tree_equal(off2.actor.params, on2.actor.params), "enabling changed the actor step"
    assert _tree_equal(off2.actor_vf.params, on2.actor_vf.params)
    # Disabled: new branch never moves. Enabled: it trains.
    assert _tree_equal(off.psi_a.params, off2.psi_a.params)
    assert _tree_equal(off.residual.params, off2.residual.params)
    assert not _tree_equal(on.psi_a.params, on2.psi_a.params), "psi_a did not train"
    assert not _tree_equal(on.residual.params, on2.residual.params), "residual did not train"
    for k in ("ac_loss", "residual_loss", "residual_spend"):
        assert math.isfinite(float(info[k])), (k, info[k])
    assert float(info["residual_spend"]) <= float(on.config["residual_eps"]) + 1e-6


def test_enabled_acting_stays_within_eps_of_decode():
    agent = _enabled_agent()
    agent = agent.replace(task_z=jnp.ones_like(agent.task_z))
    obs = jnp.zeros((6,), jnp.float32)
    a = np.asarray(agent.sample_actions(obs, seed=jax.random.PRNGKey(0)))
    assert a.shape == (2,) and np.isfinite(a).all()
    assert (np.abs(a) <= 1.0 + 1e-6).all()
    # Reconstruct the pure decode of the same latent and check the eps budget.
    noise = jax.random.normal(jax.random.PRNGKey(0), (1, 2))
    u = agent.config["u_clip"] * agent.actor(obs[None], agent.task_z[None], noise)
    pure = np.asarray(agent.decode(obs[None], u))[0]
    # Budget holds pre-clip; the final clip can only shrink the distance.
    assert (np.abs(a - pure) <= float(agent.config["residual_eps"]) + 1e-5).all()


def test_residual_does_not_touch_latent_actor():
    """residual_eps is a new-branch-only knob: with the branch enabled, changing eps
    must leave psi/phi/actor deltas byte-identical (the residual is read at stop-grad
    everywhere the shared branches could see it — which is nowhere)."""
    a1 = _enabled_agent()
    a2 = _enabled_agent(residual_eps=0.2)
    u1, _ = a1.update(_batch())
    u2, _ = a2.update(_batch())
    assert _tree_equal(u1.psi.params, u2.psi.params)
    assert _tree_equal(u1.phi.params, u2.phi.params)
    assert _tree_equal(u1.actor.params, u2.actor.params)
    assert not _tree_equal(u1.residual.params, u2.residual.params)


def test_scalar_q_pessimism_is_inert_at_zero_and_lowers_the_target_above_it():
    """P0.7: `pessimism` blends the target ensemble toward its LEAST-TASK-VALUED member.

    The previous form subtracted a per-feature spread from a vector target, which moves
    Q = psi_a^T w by -lambda * unc^T w — a quantity whose sign follows w, so on half the
    basis it RAISED the target. What is pinned here: at lambda=0 the target is exactly the
    ensemble mean (so every run to date is unaffected), and above 0 the implied target
    value Q is never above the mean-ensemble target, whatever w's signs are.
    """
    batch, agent = _batch(), _enabled_agent()
    sampled = agent.sample_step_inputs(batch, jax.random.PRNGKey(0))
    w = sampled.task_w
    a_next = agent.execute(batch["next_observations"], w, sampled.u_next)
    t_next = agent.psi_a(batch["next_observations"], w, a_next,
                         params=agent.target_psi_a)               # (P, B, z)
    q_members = (t_next * w[None]).sum(-1)                        # (P, B)

    def implied_q(lam):
        worst = jnp.argmin(q_members, axis=0)
        t_worst = jnp.take_along_axis(t_next, worst[None, :, None], axis=0)[0]
        target = (1.0 - lam) * t_next.mean(0) + lam * t_worst
        return (target * w).sum(-1)

    np.testing.assert_array_equal(np.asarray(implied_q(0.0)),
                                  np.asarray((t_next.mean(0) * w).sum(-1)))
    for lam in (0.25, 1.0):
        assert (np.asarray(implied_q(lam)) <= np.asarray(implied_q(0.0)) + 1e-6).all()


def test_pessimism_zero_matches_the_disabled_knob_step():
    """Whole-update version of the above: lambda=0 must reproduce the branch exactly."""
    a0, _ = _pessimistic_agent(0.0).update(_batch())
    ab, _ = _enabled_agent().update(_batch())
    assert _tree_equal(a0.psi_a.params, ab.psi_a.params)
    a1, _ = _pessimistic_agent(0.5).update(_batch())
    assert not _tree_equal(a1.psi_a.params, ab.psi_a.params), "pessimism knob is dead"


def test_q_spread_metric_is_logged_and_responds_to_a_flat_critic():
    """P1.4: the live 'is the action critic awake' signal. A psi_a whose output does not
    depend on the action must report ~0 relative spread; the trained one must report
    something finite. Without this the flatness was only ever found post-hoc."""
    agent = _enabled_agent()
    _, info = agent.update(_batch())
    for k in ("ac_q_spread", "ac_q_spread_rel", "ac_q_range_rel"):
        assert k in info and math.isfinite(float(info[k])), (k, info.get(k))
    assert float(info["ac_q_range_rel"]) >= float(info["ac_q_spread_rel"])

    # Zero out psi_a's action pathway by zeroing the whole final layer: Q_a becomes
    # constant in a, so the spread must collapse.
    flat = jax.tree_util.tree_map(jnp.zeros_like, agent.psi_a.params)
    spread = agent.replace(psi_a=agent.psi_a.replace(params=flat)).action_critic_spread(
        _batch(), agent.sample_step_inputs(_batch(), jax.random.PRNGKey(0)),
        jax.random.PRNGKey(1))
    assert float(spread["ac_q_spread"]) < 1e-6


def test_restore_tolerates_checkpoints_predating_the_branch(tmp_path):
    """P1.3 blocker: every psmflow checkpoint written before the action branch existed
    has no psi_a/residual entry, and flax refuses a state dict missing a target field —
    so the owed hpmatch evals could not even load. Restoring such a checkpoint must work
    and must restore the shared branches exactly."""
    import pickle

    import flax
    from utils.flax_utils import restore_agent

    trained, _ = _agent().update(_batch())
    state = flax.serialization.to_state_dict(trained)
    for k in ("psi_a", "target_psi_a", "residual"):     # strip, as an old ckpt would be
        state.pop(k)
    d = tmp_path / "run"
    d.mkdir()
    with open(d / "params_500000.pkl", "wb") as f:
        pickle.dump({"agent": state}, f)

    fresh = _agent()
    restored = restore_agent(fresh, str(d), 500000)
    assert _tree_equal(restored.psi.params, trained.psi.params)
    assert _tree_equal(restored.actor.params, trained.actor.params)
    assert _tree_equal(restored.psi_a.params, fresh.psi_a.params)   # left as initialised


def test_qa_rank_acting_stays_on_the_decode_manifold_and_uses_the_ranking():
    """P1.2 lambda-rank eval variant: rank K decoded candidates by Q_a, execute the
    argmax, NO residual. Pins that (a) the executed action is exactly one of the pure
    decodes (no residual leaks in), and (b) it is the one Q_a scores highest."""
    agent = _agent(action_critic=dict(_AC, eval_rank_k=6))
    agent = agent.replace(task_z=jnp.ones_like(agent.task_z))
    obs, key = jnp.zeros((6,), jnp.float32), jax.random.PRNGKey(3)
    a = np.asarray(agent.sample_actions(obs, seed=key))

    K = 6
    obs_b = jnp.broadcast_to(obs, (K, 6))
    w_b = jnp.broadcast_to(agent.task_z, (K, agent.task_z.shape[0]))
    u = agent.config["u_clip"] * agent.actor(obs_b, w_b, jax.random.normal(key, (K, 2)))
    cand = np.asarray(agent.decode(obs_b, u))
    Q = np.asarray((agent.psi_a(obs_b, w_b, jnp.asarray(cand)) * w_b).sum(-1).mean(0))
    np.testing.assert_allclose(a, cand[int(np.argmax(Q))], rtol=1e-6, atol=1e-6)
    # And it differs from the residual acting path on the same checkpoint (else the
    # ablation would be measuring nothing).
    pushed = _enabled_agent(residual_eps=0.2).replace(
        task_z=jnp.ones_like(agent.task_z))
    assert not np.allclose(a, np.asarray(pushed.sample_actions(obs, seed=key)))


# ---------------- P2: the FB graft (unshared basis for the action branch) -------------

def _graft_agent(**kw):
    return _agent(action_critic=dict(_AC, fb_graft=True), **kw)


def test_graft_disabled_is_byte_identical_everywhere():
    """The gate every new branch has to pass: with fb_graft off, nothing moves — not the
    shared measure, not the actor, not the pre-existing action branch, not B_a, and not
    the rng stream (the graft's own draws are folded out of it, never split from it)."""
    off = _enabled_agent()
    off2, info = off.update(_batch())
    assert _tree_equal(off.phi_a.params, off2.phi_a.params), "B_a trained with graft off"
    assert _tree_equal(off.target_phi_a, off2.target_phi_a)
    base = _enabled_agent()          # same construction, independent instance
    base2, _ = base.update(_batch())
    assert _tree_equal(off2.psi.params, base2.psi.params)
    assert _tree_equal(off2.psi_a.params, base2.psi_a.params)
    assert bool(jnp.array_equal(off2.rng, base2.rng))
    # w_a is exactly w when the graft is off, so the branch sees what it always saw.
    s = off.sample_step_inputs(_batch(), jax.random.PRNGKey(0))
    np.testing.assert_array_equal(np.asarray(s.task_w), np.asarray(s.task_w_a))


def test_graft_trains_its_own_basis_and_leaves_the_shared_one_alone():
    a = _graft_agent()
    b = _enabled_agent()
    a2, info = a.update(_batch())
    b2, _ = b.update(_batch())
    # B_a and psi_a train; the shared branches take EXACTLY the step they take without
    # the graft (identical construction, identical rng, no shared gradient).
    assert not _tree_equal(a.phi_a.params, a2.phi_a.params), "B_a did not train"
    assert not _tree_equal(a.psi_a.params, a2.psi_a.params)
    assert _tree_equal(a2.phi.params, b2.phi.params), "graft moved the shared phi"
    assert _tree_equal(a2.psi.params, b2.psi.params), "graft moved the shared psi"
    assert _tree_equal(a2.actor.params, b2.actor.params)
    for k in ("ac_loss", "ac_fb_diag", "ac_fb_offdiag", "ac_orth_loss"):
        assert math.isfinite(float(info[k])), (k, info[k])


def test_graft_loss_takes_no_gradient_from_the_shared_basis_and_vice_versa():
    """Explicit both-directions isolation: d(graft loss)/d(phi) == 0 and
    d(measure loss)/d(phi_a) == 0. This is what 'unshare the basis' has to mean."""
    a = _graft_agent()
    batch = _batch()
    sampled = a.sample_step_inputs(batch, jax.random.PRNGKey(0))

    def graft_of_phi(phi_p):
        loss, _ = a.replace(phi=a.phi.replace(params=phi_p)).action_critic_fb_loss(
            batch, sampled, a.psi_a.params, a.phi_a.params)
        return loss

    def measure_of_phi_a(phi_a_p):
        loss, _ = a.replace(phi_a=a.phi_a.replace(params=phi_a_p)).measure_loss(
            batch, sampled, a.phi.params, a.psi.params)
        return loss

    for g in jax.tree_util.tree_leaves(jax.grad(graft_of_phi)(a.phi.params)):
        assert float(jnp.abs(g).max()) == 0.0, "graft loss depends on the shared phi"
    for g in jax.tree_util.tree_leaves(jax.grad(measure_of_phi_a)(a.phi_a.params)):
        assert float(jnp.abs(g).max()) == 0.0, "the shared measure depends on B_a"


def test_graft_infers_w_a_from_its_own_basis():
    """w_a = E[r B_a(s')] must differ from w = E[r phi(s')] under the graft, and must be
    the SAME object without it (so the non-graft acting path is unchanged)."""
    rng = np.random.default_rng(0)
    nobs = rng.standard_normal((64, 6)).astype(np.float32)
    rew = rng.standard_normal(64).astype(np.float32)
    g = _graft_agent().infer_eval_z(nobs, rew)
    assert not np.allclose(np.asarray(g.task_z), np.asarray(g.task_z_a))
    n = _enabled_agent().infer_eval_z(nobs, rew)
    np.testing.assert_array_equal(np.asarray(n.task_z), np.asarray(n.task_z_a))


def test_graft_acting_still_respects_the_eps_budget():
    agent = _graft_agent()
    rng = np.random.default_rng(1)
    agent = agent.infer_eval_z(rng.standard_normal((32, 6)).astype(np.float32),
                               rng.standard_normal(32).astype(np.float32))
    obs = jnp.zeros((6,), jnp.float32)
    a = np.asarray(agent.sample_actions(obs, seed=jax.random.PRNGKey(0)))
    noise = jax.random.normal(jax.random.PRNGKey(0), (1, 2))
    u = agent.config["u_clip"] * agent.actor(obs[None], agent.task_z[None], noise)
    pure = np.asarray(agent.decode(obs[None], u))[0]
    assert (np.abs(a - pure) <= float(agent.config["residual_eps"]) + 1e-5).all()
    assert np.isfinite(a).all() and (np.abs(a) <= 1.0 + 1e-6).all()
