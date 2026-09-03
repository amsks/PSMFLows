"""Per-task offline RL in the flow's latent space -- the ceiling probe (P2).

NOT a zero-shot method and not meant to be one. Ordinary per-task offline RL over the
preimage npz's latents: scalar Q(s, u) trained on the actual task reward, an actor emitting
a latent, executed action = the frozen flow's decode. No psi, no phi, no task vector.

It answers whether the latent action space supports a working improvement loop at all:
near FQL (0.949) means the latent space and frozen flow are innocent; a low cap means the
support constraint itself costs the performance.
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

    def critic_loss(self, batch, a_data, a_next, critic_params):
        """Q over EXECUTED ACTIONS, not latents.

        The residual head only receives gradient if the critic can see its effect, so the
        critic scores actions and the dataset's own action is the in-sample anchor.
        """
        c = self.config
        q_pred = self.critic(batch["observations"], a_data, params=critic_params)
        nq = self.critic(batch["next_observations"], a_next, params=self.target_critic)
        qmean, qunc = targets_uncertainty(nq, c["num_parallel"])
        next_q = qmean - c["pessimism_penalty"] * qunc
        target = batch["rewards"] + c["discount"] * batch["masks"] * next_q
        loss = jnp.mean((q_pred - jax.lax.stop_gradient(target)) ** 2)
        return loss, {"critic_loss": loss, "q_mean": jnp.mean(q_pred),
                      "target_mean": jnp.mean(target)}

    def actor_loss(self, batch, u_data, noise, actor_params, residual_params):
        """-Q with a behaviour anchor to the dataset latent (FQL's recipe, in latent space).

        The actor and the residual head are trained by the same -Q term; the BC anchor
        stays on the LATENT, so the residual is disciplined only by its eps budget.
        """
        c = self.config
        u_a = self.actor_latent(batch["observations"], noise, params=actor_params)
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
        a_next = self.execute(batch["next_observations"], u_next)

        (_, cinfo), gc = jax.value_and_grad(self.critic_loss, argnums=3, has_aux=True)(
            batch, batch["actions"], jax.lax.stop_gradient(a_next), self.critic.params)
        critic = self.critic.apply_gradients(grads=gc)
        target_critic = polyak_update(critic.params, self.target_critic, c["tau"])

        (_, ainfo), (ga, gr) = jax.value_and_grad(
            self.actor_loss, argnums=(3, 4), has_aux=True)(
            batch, u_data, jax.random.normal(r_act, (B, d_a)),
            self.actor.params, self.residual.params)
        actor = self.actor.apply_gradients(grads=ga)
        residual = self.residual.apply_gradients(grads=gr)
        return (self.replace(rng=rng, critic=critic, target_critic=target_critic,
                             actor=actor, residual=residual), {**cinfo, **ainfo})

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

        critic_def = Value(hidden_dims=tuple(config["value_hidden_dims"]),
                           layer_norm=config["layer_norm"],
                           num_ensembles=config["num_parallel"])
        critic = TrainState.create(
            critic_def, critic_def.init(rc, ex_observations, ex_actions)["params"],
            tx=optax.adam(config["lr"]))

        actor_def = NoiseConditionedActor(
            action_dim=action_dim, hidden_dim=config["actor"]["hidden_dim"],
            hidden_layers=config["actor"]["hidden_layers"],
            embedding_layers=config["actor"]["embedding_layers"])
        actor = TrainState.create(
            actor_def, actor_def.init(ra, ex_observations, ex_z, ex_u)["params"],
            tx=optax.adam(config["lr"]))

        # W4 residual head: (s, u) -> per-dim correction, scaled by eps and tanh-bounded.
        # NoiseConditionedActor already tanh-bounds its output, so it is delta post-tanh;
        # `execute` only scales it by eps.
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
