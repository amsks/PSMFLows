"""CRL + FB FlowBC actor (JAX). Subclasses vendored CRLAgent: keeps the
contrastive critic, replaces the actor with FB's flow-matching velocity field
+ one-shot noise-conditioned actor (Q-guidance from CRL's min(q1,q2)).
Registered into agents.agents['crl_flowbc'] by sitecustomize. Never edits OGBench.
"""
import flax
import jax
import jax.numpy as jnp
from agents.crl import CRLAgent, get_config as _crl_get_config
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import GCBilinearValue
import optax

from ogbench_flow import GCNoiseActor, GCVectorField


class CRLFlowBCAgent(CRLAgent):
    """CRL with FB's FlowBC actor. Inherits contrastive_loss + update."""

    def _flow_rollout(self, obs, noise, params):
        fs = self.config['flow_steps']
        actions = noise
        for i in range(fs):
            t = jnp.ones((noise.shape[0], 1)) * (i / fs)
            vels = self.network.select('actor_vf')(obs, actions, t, params=params)
            actions = actions + vels / fs
        return jnp.clip(actions, -1.0, 1.0)

    def actor_loss(self, batch, grad_params, rng=None):
        obs, goals, actions = batch['observations'], batch['actor_goals'], batch['actions']
        rng = rng if rng is not None else self.rng
        rng, x0_rng, t_rng, n_rng = jax.random.split(rng, 4)

        # 1) flow-matching loss trains actor_vf
        x_1 = actions
        x_0 = jax.random.normal(x0_rng, actions.shape)
        t = jax.random.uniform(t_rng, (actions.shape[0], 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.network.select('actor_vf')(obs, x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # 2) Q-max on one-shot actor's actions (CRL contrastive Q).
        # The critic is evaluated at its FROZEN params (no params=grad_params), as
        # in OGBench's DDPG+BC actor loss (agents/crl.py): otherwise -q.mean()
        # backprops into the critic and the single optimizer inflates Q instead of
        # improving the actor -> contrastive critic collapses (q explodes, logits
        # pos==neg, categorical_accuracy stuck at chance, 0% eval success).
        noise = jax.random.normal(n_rng, actions.shape)
        actor_actions = self.network.select('actor')(obs, goals, noise, params=grad_params)
        q1, q2 = self.network.select('critic')(obs, goals, actor_actions)
        q = jnp.minimum(q1, q2)
        actor_loss = -q.mean()

        # 3) BC-distillation toward frozen flow rollout (FB normalization)
        bc_loss = 0.0
        bc_error = 0.0
        if self.config['bc_coeff'] > 0:
            target = self._flow_rollout(obs, noise, jax.lax.stop_gradient(grad_params))
            bc_error = jnp.mean((actor_actions - target) ** 2)
            bc_loss = self.config['bc_coeff'] * bc_error
            actor_loss = actor_loss / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6) + bc_loss
        actor_loss = actor_loss + bc_flow_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'bc_error': bc_error,
            'q_mean': q.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        critic_loss, critic_info = self.contrastive_loss(batch, grad_params, 'critic')
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v
        return critic_loss + actor_loss, info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        del temperature  # flow actor is noise-driven
        # Noise must share the obs batch shape. Eval passes a single, unbatched
        # observation (evaluation.py:80); training/tests pass batches. The obs
        # event rank is 1 for state and 3 (H,W,C) for pixels (encoder present;
        # see encoders.py:96), so strip the event dims to get the batch shape.
        event_rank = 1 if self.config['encoder'] is None else 3
        batch_shape = observations.shape[:-event_rank]
        noise = jax.random.normal(seed, (*batch_shape, self.config['action_dim']))
        actions = self.network.select('actor')(observations, goals, noise)
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]
        config = dict(config)
        config['action_dim'] = action_dim

        encoders = {}
        if config['encoder'] is not None:
            em = encoder_modules[config['encoder']]
            encoders['critic_state'] = em()
            encoders['critic_goal'] = em()
            encoders['actor_obs'] = em()
            encoders['actor_goal'] = em()

        critic_def = GCBilinearValue(
            hidden_dims=config['value_hidden_dims'],
            latent_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            value_exp=False,
            state_encoder=encoders.get('critic_state'),
            goal_encoder=encoders.get('critic_goal'),
        )
        actor_vf_def = GCVectorField(
            hidden_dim=config['actor_vf_hidden_dim'],
            hidden_layers=config['actor_vf_hidden_layers'],
            obs_encoder=encoders.get('actor_obs'),
        )
        actor_def = GCNoiseActor(
            hidden_dim=config['actor_hidden_dim'],
            hidden_layers=config['actor_hidden_layers'],
            embedding_layers=config['actor_embedding_layers'],
            obs_encoder=encoders.get('actor_obs'),
            goal_encoder=encoders.get('actor_goal'),
        )

        ex_noise = jnp.zeros_like(ex_actions)
        ex_t = jnp.zeros((ex_actions.shape[0], 1))
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals, ex_noise)),
            actor_vf=(actor_vf_def, (ex_observations, ex_actions, ex_t)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr_actor_vf'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = _crl_get_config()
    with config.unlocked():
        config.agent_name = 'crl_flowbc'
        config.flow_steps = 10
        config.bc_coeff = 3.0
        config.lr_actor_vf = 3e-4
        config.actor_vf_hidden_dim = 512
        config.actor_vf_hidden_layers = 4
        config.actor_hidden_dim = 512
        config.actor_hidden_layers = 2
        config.actor_embedding_layers = 2
    return config
