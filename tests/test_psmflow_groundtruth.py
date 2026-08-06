"""Ground-truth chain-MDP test for PSMFlow: validates the MATH, not the plumbing.

Every other psmflow test checks shapes, finiteness, and wiring — all of which a subtly
wrong loss would still satisfy. This one puts the agent on an MDP whose occupancies are
known by construction and asks whether the learned psi^T phi and the amortized actor
actually solve it.

Chain MDP: 5 states on a line, obs = one-hot(5), d_a = 1. `a > 0` moves right, `a < 0`
moves left, clipped at the ends. Behaviour data covers both actions from every state.
Latents are the identity (`noise_preimage := actions`), so the frozen flow is UNUSED here
and the u <-> a correspondence is exact — which is what makes the ground truth knowable.

Under the 08-05 semantics (decisions.tex) psi(s, w, u)^T phi is the occupancy of "emit
latent u now, then follow the actor's policy for w". With reward at state 4: the
goal-ward latent must outscore the goal-averse one from every state, value must increase
toward the goal, and — the policy-improvement pin — the ACTOR itself must emit goal-ward
latents under the inferred w even though the behaviour data is direction-balanced.
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


def _trained_agent(steps=2000):
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
    for _ in range(steps):
        idx = rng.integers(0, n, size=64)
        agent, _ = agent.update({k: v[idx] for k, v in data.items()})
    # Reward: 1 at state 4 (goal). Infer w on the dataset's next states.
    rewards = data["next_observations"][:, S - 1].astype(np.float32)
    return agent.infer_eval_z(data["next_observations"], rewards), data


def _score(agent, s, u_sign):
    """psi(s, w, u)^T w for the inferred task w and a single latent u."""
    obs = jnp.eye(S)[s].astype(jnp.float32)[None]
    u = jnp.full((1, 1), 0.8 * u_sign)
    qpsi = agent.psi(obs, agent.task_z[None], u)         # (P, 1, z_dim)
    return float((qpsi * agent.task_z).sum(-1).mean(axis=0)[0])


def test_chain_q_prefers_goalward_latent():
    agent, _ = _trained_agent()
    # (i) goal-ward latent beats goal-averse latent at every non-goal state
    for s in range(S - 1):
        assert _score(agent, s, +1) > _score(agent, s, -1), f"state {s}: rightward not preferred"
    # (ii) value (best score) increases toward the goal
    vals = [max(_score(agent, s, +1), _score(agent, s, -1)) for s in range(S)]
    assert vals[3] > vals[0], f"value not increasing toward goal: {vals}"


def test_chain_actor_emits_goalward_latents():
    """The policy-improvement pin. The behaviour data is direction-balanced (both
    actions from every state), so BC alone gives a sign-ambiguous actor; only the Q term
    can break the tie toward the goal. Under the inferred w the actor's latent must be
    positive (rightward) at every non-goal state — this is exactly what the 08-05
    redesign buys, and what no fixed-u family can express."""
    agent, _ = _trained_agent()
    noise = jnp.zeros((1, 1))  # deterministic head; the tie-break must come from Q
    for s in range(S - 1):
        obs = jnp.eye(S)[s].astype(jnp.float32)[None]
        u = float(agent.config["u_clip"] * agent.actor(obs, agent.task_z[None], noise)[0, 0])
        assert u > 0, f"state {s}: actor latent {u:.3f} is not goal-ward"
