"""Per-task offline RL in the flow's latent space -- the ceiling probe (P2).

This is NOT a zero-shot method and is not meant to be one. The preimage npz already turns
the dataset into a latent-action dataset (u* per transition), so we can run ordinary
per-task offline RL over latents: a scalar critic Q(s, u) trained by TD on the ACTUAL task
reward, an actor emitting a latent, and the executed action being the frozen flow's decode
of that latent. No psi, no phi, no task vector w.

What it answers: can the latent action space support a working improvement loop at all?

  lands near FQL (0.949)  -> the latent space and frozen flow are innocent; whatever is
                             wrong is in the zero-shot representation, and transplanting
                             our actor into a working loop (LatentFB) has headroom.
  caps low (~0.3)         -> the support constraint itself costs the performance; a better
                             representation cannot exceed this, and LatentFB would have
                             capped here too.

Deliberately the smallest thing that answers the question: FQL's critic recipe with the
action replaced by the latent, and psmflow's frozen decode reused verbatim.

`critic_input` (2026-09-03) selects WHAT THE CRITIC SCORES, and the two settings are two
different algorithms sharing one acting path:

  action  (default, and what every earlier run did) -- Q(s, a) over the EXECUTED action.
          The actor gradient reaches the latent THROUGH the frozen decoder, and the
          residual head is meaningful because the critic can see it.
  latent  -- Q(s, u). The offline variant of DSRL (Wagenmaker et al. 2025), which steers a
          frozen generative policy through its input noise. Its offline form needs noise
          aliasing (offline data has actions but no noise); Stage-B preimages ARE that
          aliasing, so `u_data = clip(noise_preimage)` is the in-sample TD anchor. The
          frozen flow is never called during training -- no decode in the critic, none in
          the actor loss -- and reappears only in `sample_actions`. residual_eps must be 0.

`action` is byte-for-byte the pre-switch computation: same rng splits, same shapes, same
logged values (tests/test_latentrl_smoke.py pins them).
"""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from agents.psm import _plain_config, polyak_update, targets_uncertainty
from agents.psmflow import _load_flow_params
from utils.flax_utils import TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value
from utils.psm_networks import NoiseConditionedActor


class LatentRLAgent(flax.struct.PyTreeNode):
    rng: Any
    critic: TrainState
    target_critic: Any
    actor: TrainState
    residual: TrainState   # W4 bounded residual head; inert at residual_eps=0
    flow_vf: Any        # FROZEN
    flow_onestep: Any   # FROZEN
    config: Any = nonpytree_field()
    flow_vf_def: Any = nonpytree_field(default=None)
    flow_onestep_def: Any = nonpytree_field(default=None)

    # `w` is a zero vector of width zero-cost: NoiseConditionedActor takes (s, z, noise),
    # and passing a constant z makes it a plain (s, noise) actor without a second network.
    def _z(self, n):
        return jnp.zeros((n, self.config["z_dim"]))

    def actor_latent(self, obs, noise, params=None):
        p = self.actor.params if params is None else params
        return self.config["u_clip"] * self.actor(
            obs, self._z(obs.shape[0]), noise, params=p)

    def decode(self, observations, u):
        """Frozen one-step decode, identical to psmflow's acting path."""
        a = self.flow_onestep_def.apply({"params": self.flow_onestep}, observations, u)
        return jnp.clip(a, -1.0, 1.0)

    def execute(self, observations, u, residual_params=None):
        """The action actually taken: decode plus a bounded residual.

            a = clip(G(s, u) + eps * delta(s, u)),   delta tanh-bounded to [-1, 1]

        eps is a hard per-dimension budget, so the executed action is never further than
        eps from something the frozen flow could have produced. C1 measured that the
        task-optimal actions sit off the behaviour support, so this is the dial that buys
        support violation back in a measured quantity rather than all-or-nothing.

        At residual_eps=0 the head is inert (its output is multiplied by zero) but the
        CRITIC still scores executed actions rather than latents -- see critic_loss. That
        makes eps=0 its own anchor arm, not a reproduction of the P2 runs.
        """
        a = self.decode(observations, u)
        eps = self.config["residual_eps"]
        if eps > 0.0:
            p = self.residual.params if residual_params is None else residual_params
            a = a + eps * self.residual(observations, self._z(observations.shape[0]),
                                        u, params=p)
        return jnp.clip(a, -1.0, 1.0)

    def critic_loss(self, batch, x_data, x_next, critic_params):
        """One TD loss for both critic inputs; `update` decides what x is.

        critic_input='action': x is the EXECUTED action, and the dataset's own action is
        the in-sample anchor -- the residual head only receives gradient if the critic can
        see its effect. critic_input='latent': x is the LATENT, and the anchor is the
        Stage-B preimage u*. Reward, mask, discount and the pessimism-shrunk bootstrap are
        identical either way; only the argument changes.
        """
        c = self.config
        q_pred = self.critic(batch["observations"], x_data, params=critic_params)
        nq = self.critic(batch["next_observations"], x_next, params=self.target_critic)
        qmean, qunc = targets_uncertainty(nq, c["num_parallel"])
        next_q = qmean - c["pessimism_penalty"] * qunc
        target = batch["rewards"] + c["discount"] * batch["masks"] * next_q
        loss = jnp.mean((q_pred - jax.lax.stop_gradient(target)) ** 2)
        info = {"critic_loss": loss, "q_mean": jnp.mean(q_pred),
                "target_mean": jnp.mean(target)}
        if c["critic_input"] == "latent":
            # Same number as q_mean, under the name that says what it is scoring: mean Q
            # on the in-sample latents. Kept out of the action path so that path's logged
            # dict is exactly what it was.
            info["critic_q_data"] = jnp.mean(q_pred)
        return loss, info

    def latent_q_spread(self, batch, key):
        """"Is the latent critic awake?" -- psmflow's ac_q_spread_rel, for Q(s, u).

        16 clipped prior latents at the same states; if Q barely moves across them the
        -Q gradient carries no direction and the actor is BC plus noise. D3 measured 1.1%
        relative spread for the measure head, which is what made that arm hopeless; this
        makes the same number visible live instead of post-hoc on a finished checkpoint.
        """
        c = self.config
        n = min(64, batch["observations"].shape[0])
        obs = batch["observations"][:n]
        u = jnp.clip(jax.random.normal(key, (16, n, c["action_dim"])),
                     -c["u_clip"], c["u_clip"])

        def q_of(u_k):
            qmean, _ = targets_uncertainty(self.critic(obs, u_k), c["num_parallel"])
            return qmean                                                    # (n,)

        Q = jax.vmap(q_of)(u)                                               # (16, n)
        spread = Q.std(0).mean()
        return {"q_spread": spread, "q_spread_rel": spread / (jnp.abs(Q).mean() + 1e-8)}

    def actor_loss(self, batch, u_data, noise, actor_params, residual_params):
        """-Q with a behaviour anchor to the dataset latent (FQL's recipe, in latent space).

        The actor and the residual head are trained by the same -Q term; the BC anchor
        stays on the LATENT, so the residual is disciplined only by its eps budget.
        """
        c = self.config
        u_a = self.actor_latent(batch["observations"], noise, params=actor_params)
        if c["critic_input"] == "latent":
            # DSRL: the critic scores the latent, so there is nothing to decode. The
            # frozen flow does not appear in this graph at all -- pinned by a test that
            # perturbs flow_onestep and requires this loss not to move.
            qs = self.critic(batch["observations"], u_a)
            qmean, qunc = targets_uncertainty(qs, c["num_parallel"])
            q = qmean - c["actor_pessimism_penalty"] * qunc
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-8)
            bc = jnp.mean((u_a - u_data) ** 2)
            loss = q_loss + c["bc_alpha_latent"] * bc
            return loss, {"actor_loss": loss, "actor_q": q.mean(), "bc_loss": bc}
        a_exec = self.execute(batch["observations"], u_a, residual_params=residual_params)
        qs = self.critic(batch["observations"], a_exec)
        qmean, qunc = targets_uncertainty(qs, c["num_parallel"])
        q = qmean - c["actor_pessimism_penalty"] * qunc
        # Normalised as in psm/psmflow so alpha is scale-free across environments.
        q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-8)
        bc = jnp.mean((u_a - u_data) ** 2)
        loss = q_loss + c["alpha"] * bc
        # How far the executed action strays from the pure decode: the support budget
        # actually spent, which is the x-axis of the tradeoff curve.
        spend = jnp.mean(jnp.abs(a_exec - self.decode(batch["observations"], u_a)))
        return loss, {"actor_loss": loss, "actor_q": q.mean(), "actor_bc": bc,
                      "residual_spend": spend}

    @jax.jit
    def update(self, batch):
        c = self.config
        rng, r_next, r_act = jax.random.split(self.rng, 3)
        B, d_a = batch["observations"].shape[0], c["action_dim"]
        u_data = jnp.clip(jnp.asarray(batch["noise_preimage"]), -c["u_clip"], c["u_clip"])
        u_next = self.actor_latent(batch["next_observations"],
                                   jax.random.normal(r_next, (B, d_a)))
        # The critic's two arguments, per critic_input. The latent branch never calls
        # `execute`, so the frozen decoder is absent from the whole training graph; the
        # action branch is exactly the pre-2026-09-03 pair.
        if c["critic_input"] == "latent":
            x_data, x_next = u_data, u_next
        else:
            x_data, x_next = batch["actions"], self.execute(batch["next_observations"], u_next)

        (_, cinfo), gc = jax.value_and_grad(self.critic_loss, argnums=3, has_aux=True)(
            batch, x_data, jax.lax.stop_gradient(x_next), self.critic.params)
        critic = self.critic.apply_gradients(grads=gc)
        target_critic = polyak_update(critic.params, self.target_critic, c["tau"])

        (_, ainfo), (ga, gr) = jax.value_and_grad(
            self.actor_loss, argnums=(3, 4), has_aux=True)(
            batch, u_data, jax.random.normal(r_act, (B, d_a)),
            self.actor.params, self.residual.params)
        actor = self.actor.apply_gradients(grads=ga)
        # Under critic_input=latent the residual is not in the loss, so `gr` is all zeros
        # and adam's update is exactly zero -- the head stays at init, inert, as asserted.
        residual = self.residual.apply_gradients(grads=gr)
        info = {**cinfo, **ainfo}
        if c["critic_input"] == "latent":
            # Key FOLDED out of self.rng rather than split from it, so adding the
            # diagnostic leaves the actor/critic draws above untouched.
            info.update(self.latent_q_spread(batch, jax.random.fold_in(self.rng, 137)))
        return (self.replace(rng=rng, critic=critic, target_critic=target_critic,
                             actor=actor, residual=residual), info)

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        seed = self.rng if seed is None else seed
        noise = jax.random.normal(seed, (1, self.config["action_dim"]))
        u = self.actor_latent(observations[None], noise)
        return self.execute(observations[None], u)[0]

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, rc, ra, rr = jax.random.split(rng, 4)
        action_dim = ex_actions.shape[-1]
        config["action_dim"] = action_dim
        config["ob_dims"] = ex_observations.shape[-1]
        ex_u = jnp.zeros((ex_observations.shape[0], action_dim))
        ex_z = jnp.zeros((ex_observations.shape[0], config["z_dim"]))

        critic_input = config.get("critic_input", "action")
        assert critic_input in ("action", "latent"), critic_input
        if critic_input == "latent":
            assert float(config["residual_eps"]) == 0.0, (
                "critic_input=latent requires residual_eps=0.0: the critic scores latents, "
                f"so it cannot see a residual, and residual_eps={config['residual_eps']} "
                "would put an unscored correction into the executed action")
        if "critic_input" not in config or "bc_alpha_latent" not in config:
            # A config restored from a flags.json written before 2026-09-03 has neither
            # key; fill them with the values that reproduce that run.
            with config.unlocked():
                config["critic_input"] = critic_input
                config["bc_alpha_latent"] = config.get("bc_alpha_latent", config["alpha"])

        critic_def = Value(hidden_dims=tuple(config["value_hidden_dims"]),
                           layer_norm=config["layer_norm"],
                           num_ensembles=config["num_parallel"])
        # Both inputs have width action_dim, so the ensemble is the same shape either way;
        # initialising on the argument the critic will actually receive keeps that explicit.
        critic_ex = ex_u if critic_input == "latent" else ex_actions
        critic = TrainState.create(
            critic_def, critic_def.init(rc, ex_observations, critic_ex)["params"],
            tx=optax.adam(config["lr"]))

        actor_def = NoiseConditionedActor(
            action_dim=action_dim, hidden_dim=config["actor"]["hidden_dim"],
            hidden_layers=config["actor"]["hidden_layers"],
            embedding_layers=config["actor"]["embedding_layers"])
        actor = TrainState.create(
            actor_def, actor_def.init(ra, ex_observations, ex_z, ex_u)["params"],
            tx=optax.adam(config["lr"]))

        # W4 residual head: (s, u) -> per-dim correction, scaled by eps and tanh-bounded.
        # NoiseConditionedActor already tanh-bounds its output, so it IS delta with the
        # tanh applied; `execute` therefore scales it by eps without tanh-ing again.
        residual_def = NoiseConditionedActor(
            action_dim=action_dim, hidden_dim=config["residual"]["hidden_dim"],
            hidden_layers=config["residual"]["hidden_layers"],
            embedding_layers=config["residual"]["embedding_layers"])
        residual = TrainState.create(
            residual_def, residual_def.init(rr, ex_observations, ex_z, ex_u)["params"],
            tx=optax.adam(config["lr"]))

        # Frozen behaviour flow, exactly as psmflow builds it: defs rebuilt from config,
        # params restored from the Stage-A checkpoint.
        flow_hidden = tuple(config["flow"]["hidden_dims"])
        vf_def = ActorVectorField(hidden_dims=flow_hidden, action_dim=action_dim,
                                  layer_norm=config["flow"]["layer_norm"])
        onestep_def = ActorVectorField(hidden_dims=flow_hidden, action_dim=action_dim,
                                       layer_norm=config["flow"]["layer_norm"])
        assert config.get("flow_ckpt_path"), (
            "latentrl needs agent.flow_ckpt_path: the whole point is to act through the "
            "SAME frozen flow psmflow decodes through")
        vf, onestep = _load_flow_params(
            config["flow_ckpt_path"], config["flow_ckpt_epoch"], config,
            ex_observations, ex_actions)

        import copy
        cfg = flax.core.FrozenDict(_plain_config(config))
        return cls(rng=rng, critic=critic, target_critic=copy.deepcopy(critic.params),
                   actor=actor, residual=residual,
                   flow_vf=vf, flow_onestep=onestep, config=cfg,
                   flow_vf_def=vf_def, flow_onestep_def=onestep_def)
