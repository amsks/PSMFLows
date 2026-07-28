"""Ground-truth chain-MDP test for PSMFlow: validates the MATH, not the plumbing.

Every other psmflow test checks shapes, finiteness, and wiring — all of which a subtly
wrong loss would still satisfy. This one puts the agent on an MDP whose occupancies are
known by construction and asks whether the learned psi^T phi actually ranks latent
policies correctly.

Chain MDP: 5 states on a line, obs = one-hot(5), d_a = 1. `a > 0` moves right, `a < 0`
moves left, clipped at the ends. Behaviour data covers both actions from every state.
Latents are the identity (`noise_preimage := actions`), so the frozen flow is UNUSED here
and the u <-> a correspondence is exact — which is what makes the ground truth knowable.
Under a fixed positive latent the policy walks right and absorbs at state 4; with reward
at state 4, the goal-ward latent must score higher from every state, and value must
increase toward the goal.
"""
import jax
import jax.numpy as jnp
import numpy as np

from agents.psmflow import PSMFlowAgent, get_config

S = 5


def _chain_dataset(reps=40):
    """All (state, action in {-0.8, +0.8}) transitions, tiled `reps` times."""
    obs, act, nxt = [], [], []
    for s in range(S):
        for a in (-0.8, 0.8):
            s2 = int(np.clip(s + (1 if a > 0 else -1), 0, S - 1))
            obs.append(np.eye(S, dtype=np.float32)[s])
            act.append([a])
            nxt.append(np.eye(S, dtype=np.float32)[s2])
    obs = np.tile(np.array(obs, np.float32), (reps, 1))
    act = np.tile(np.array(act, np.float32), (reps, 1))
    nxt = np.tile(np.array(nxt, np.float32), (reps, 1))
    return dict(observations=obs, actions=act, next_observations=nxt,
                noise_preimage=act.copy())  # identity latents: u := a


def _score(agent, s, u_sign):
    """max_u' psi(s, u', u)^T w for a single candidate latent u."""
    c = agent.config
    Mi = 32
    obs = jnp.broadcast_to(jnp.eye(S)[s].astype(jnp.float32), (Mi, S))
    u = jnp.full((Mi, 1), 0.8 * u_sign)
    u_idx = jnp.clip(jax.random.normal(jax.random.PRNGKey(7), (Mi, 1)), -c["u_clip"], c["u_clip"])
    qpsis = agent.psi(obs, u_idx, u)
    Qs = (qpsis * agent.task_z).sum(-1).mean(axis=0)  # ensemble mean, (Mi,)
    return float(Qs.max())


def test_chain_gpi_prefers_goalward_latent():
    cfg = get_config()
    with cfg.unlocked():
        cfg["allow_untrained_flow"] = True
        cfg["z_dim"] = 16
        cfg["batch_size"] = 64
        cfg["discount"] = 0.9
        cfg["ortho_coef"] = 1.0
        cfg["lr_phi"] = 1e-4      # tiny problem: the reference 1e-5 is needlessly slow
        cfg["sf"]["hidden_dim"] = 128
        cfg["phi"]["hidden_dim"] = 64
    data = _chain_dataset()
    n = data["observations"].shape[0]
    agent = PSMFlowAgent.create(0, data["observations"][:1], data["actions"][:1], cfg)

    rng = np.random.default_rng(0)
    for _ in range(2000):
        idx = rng.integers(0, n, size=64)
        batch = {k: v[idx] for k, v in data.items()}
        agent, _ = agent.update(batch)

    # Reward: 1 at state 4 (goal). Infer w on the dataset's next states.
    rewards = data["next_observations"][:, S - 1].astype(np.float32)
    agent = agent.infer_eval_z(data["next_observations"], rewards)

    # (i) goal-ward latent beats goal-averse latent at every state
    for s in range(S - 1):
        assert _score(agent, s, +1) > _score(agent, s, -1), f"state {s}: rightward not preferred"
    # (ii) value (best score) increases toward the goal
    vals = [max(_score(agent, s, +1), _score(agent, s, -1)) for s in range(S)]
    assert vals[3] > vals[0], f"value not increasing toward goal: {vals}"


def _score_uu(agent, s, u_sign, uprime_sign):
    """psi(s, u', u)^T w for a FIXED pair — no max over u'.

    The test above maxes over u', which marginalizes away the very distinction the
    method rests on. Holding both slots fixed is what makes the two-slot semantics
    observable.
    """
    obs = jnp.eye(S)[s].astype(jnp.float32)[None]        # (1, S)
    u = jnp.full((1, 1), 0.8 * u_sign)                   # action slot: latent taken NOW
    u_idx = jnp.full((1, 1), 0.8 * uprime_sign)          # z slot: policy followed AFTER
    qpsi = agent.psi(obs, u_idx, u)                      # (P, 1, z_dim)
    return float((qpsi * agent.task_z).sum(-1).mean(axis=0)[0])


def test_continuation_policy_dominates_the_immediate_latent():
    """The u' slot must mean "the policy followed from s' onward", not decoration.

    Ground truth on the chain with reward at state 4 and discount 0.9, writing t=1 for
    the state reached by the immediate latent u:

      (u=-0.8, u'=+0.8) at state s: step left once, then walk right and absorb at 4.
                                    Occupancy of the goal ~ gamma^k/(1-gamma), LARGE.
      (u=+0.8, u'=-0.8) at state s: step right once, then walk left to 0 and stay.
                                    Reaches the goal at most once, then never. SMALL.

    So the CONTINUATION policy must dominate the immediate action. Both orderings use
    the same two latent values, so nothing about the marginal preference for +0.8 can
    produce this — only a measure that conditions the bootstrap on u' can.

    This is the property the plan names as the prime suspect for a wiring bug, and the
    max-over-u' test above does NOT catch it: verified by mutation, both
    `tM = psi(s', u', u)` (continuation = data latent) and a u/u' slot swap in the
    online M leave that test passing.
    """
    cfg = get_config()
    with cfg.unlocked():
        cfg["allow_untrained_flow"] = True
        cfg["z_dim"] = 16
        cfg["batch_size"] = 64
        cfg["discount"] = 0.9
        cfg["ortho_coef"] = 1.0
        cfg["lr_phi"] = 1e-4
        cfg["sf"]["hidden_dim"] = 128
        cfg["phi"]["hidden_dim"] = 64
    data = _chain_dataset()
    n = data["observations"].shape[0]
    agent = PSMFlowAgent.create(0, data["observations"][:1], data["actions"][:1], cfg)

    rng = np.random.default_rng(0)
    for _ in range(2000):
        idx = rng.integers(0, n, size=64)
        agent, _ = agent.update({k: v[idx] for k, v in data.items()})

    rewards = data["next_observations"][:, S - 1].astype(np.float32)
    agent = agent.infer_eval_z(data["next_observations"], rewards)

    for s in range(S - 1):
        goalward_continuation = _score_uu(agent, s, -1, +1)   # wrong step, right policy
        goalward_step_only = _score_uu(agent, s, +1, -1)      # right step, wrong policy
        assert goalward_continuation > goalward_step_only, (
            f"state {s}: continuation policy u' does not dominate the immediate latent u "
            f"({goalward_continuation:.4f} vs {goalward_step_only:.4f}); the bootstrap is "
            "not conditioned on u'")
