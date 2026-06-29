import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
import distrax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


class FQLAgent(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _get_preimage_and_jacobian(self, state, action, n_steps):
        """Preimage of `action` and the forward flow-map Jacobian d(action)/d(noise) at it.

        Single example: `state` (ob_dim,), `action` (action_dim,). vmap for batches
        so `jax.jacfwd` gives a per-example (A, A) Jacobian, not the full (B, A, B, A).
        """
        if self.config['encoder'] is not None:
            state = self.network.select('actor_bc_flow_encoder')(state)
        action = jnp.clip(action, -1, 1)

        def flow_fn(x_t, t):
            return self.network.select('actor_bc_flow')(state, x_t, t, is_encoded=True)

        def implicit_euler_step(x_t, t):
            def body(_, x_next):
                return x_t - flow_fn(x_next, t) / n_steps

            return jax.lax.fori_loop(0, 5, body, x_t)

        def implicit_euler_loop(carry, i):
            return implicit_euler_step(carry, jnp.full((1,), i / n_steps)), None

        x_0, _ = jax.lax.scan(implicit_euler_loop, action, jnp.arange(n_steps - 1, -1, -1))

        def forward_map(noise):
            def body(x, i):
                return x + flow_fn(x, jnp.full((1,), i / n_steps)) / n_steps, None

            return jax.lax.scan(body, noise, jnp.arange(n_steps))[0]

        return x_0, jax.jacfwd(forward_map)(x_0)

    def _get_predistribution_proposal(self, state, action, n_steps, alpha=1.0):
        """Local Gaussian proposal (mean, cov) for the preimage of `action`.

        cov is the Laplace covariance of the target pi(x) ~ exp(-alpha*||flow(x)-action||)
        around the preimage: (1/alpha^2)(J^T J)^{-1}, eigenvalues clipped for stability.
        Larger alpha (lower temperature) => tighter proposal.
        """
        x_0, jacobian = self._get_preimage_and_jacobian(state, action, n_steps)
        gram = jacobian.T @ jacobian + 1e-6 * jnp.eye(jacobian.shape[-1], dtype=jacobian.dtype)
        eigvals, eigvecs = jnp.linalg.eigh(gram)
        cov_eigvals = jnp.clip(1.0 / (alpha ** 2 * eigvals), a_min=0.01, a_max=1.0)
        cov = (eigvecs * cov_eigvals[None, :]) @ eigvecs.T
        return x_0, cov
    
    def compute_full_proposal_distribution(self, state, action, rng, num_samples=100, n_steps=10, n_initial_steps=100, alpha=1.0):
        """Refine the preimage proposal toward pi(x) ~ exp(-alpha * ||flow(x) - action||) by importance sampling.

        alpha is an inverse temperature (1/T): larger alpha => sharper target.
        """
        x_0, cov = self._get_predistribution_proposal(state, action, n_initial_steps, alpha)
        state_b = jnp.broadcast_to(state, (num_samples, *state.shape))

        def _step(carry, _):
            x_0, cov, rng = carry
            prop_dist = distrax.MultivariateNormalFullCovariance(loc=x_0, covariance_matrix=cov)
            rng, sample_rng = jax.random.split(rng)
            samples, log_prob = prop_dist.sample_and_log_prob(seed=sample_rng, sample_shape=(num_samples,))
            actions = self.compute_flow_actions(state_b, noises=samples)
            dist = alpha * jnp.linalg.norm(actions - action[None], axis=-1)
            weights = jax.nn.softmax(-dist - log_prob, axis=0)
            ess = 1.0 / jnp.sum(weights ** 2)
            new_x_0 = jnp.sum(weights[..., None] * samples, axis=0)
            diff = samples - new_x_0[None, :]
            new_cov = (weights[..., None] * diff).T @ diff + 1e-6 * jnp.eye(cov.shape[-1], dtype=cov.dtype)
            return (new_x_0, new_cov, rng), ess

        (x_0, cov, rng), ess = jax.lax.scan(_step, (x_0, cov, rng), None, length=n_steps)
        return x_0, cov, ess

    def compute_full_proposal_distribution_em(self, state, action, rng, num_samples=100, n_steps=10, n_initial_steps=100, alpha=1.0, n_components=3):
        """Importance-weighted EM fit of a Gaussian mixture to pi(x) ~ exp(-alpha * ||flow(x) - action||).

        Samples are drawn from the current mixture; per-sample IS weights w_n ~ pi/q
        carry the energy, membership responsibilities r_{k,n} assign them to components,
        and the M-step uses gamma_{k,n} = w_n * r_{k,n}. alpha is an inverse temperature.
        """
        x_0, cov = self._get_predistribution_proposal(state, action, n_initial_steps, alpha)
        action_dim = x_0.shape[-1]

        rng, init_rng = jax.random.split(rng)
        means = jax.random.multivariate_normal(init_rng, mean=x_0, cov=cov, shape=(n_components,))
        covs = jnp.array([cov for _ in range(n_components)])
        weights = jnp.ones(n_components) / n_components

        def _em_step(carry, _):
            means, covs, weights, rng = carry
            rng, sample_rng = jax.random.split(rng)

            component_rng = jax.random.split(sample_rng, n_components)
            component_samples = jax.vmap(
                lambda m, c, r: distrax.MultivariateNormalFullCovariance(loc=m, covariance_matrix=c)
                    .sample(seed=r, sample_shape=(num_samples // n_components,))
            )(means, covs, component_rng)
            samples = component_samples.reshape((-1, action_dim))

            state_b = jnp.broadcast_to(state, (samples.shape[0], *state.shape))
            actions = self.compute_flow_actions(state_b, noises=samples)
            log_energy = -alpha * jnp.linalg.norm(actions - action[None], axis=-1)

            log_likelihoods = jax.vmap(
                lambda m, c: jax.vmap(
                    lambda x: distrax.MultivariateNormalFullCovariance(loc=m, covariance_matrix=c).log_prob(x)
                )(samples)
            )(means, covs)

            log_joint = jnp.log(weights[..., None]) + log_likelihoods
            log_q = jax.scipy.special.logsumexp(log_joint, axis=0)
            responsibilities = jnp.exp(log_joint - log_q[None, :])
            sample_weights = jax.nn.softmax(log_energy - log_q, axis=0)
            gamma = responsibilities * sample_weights[None, :]

            n_k = jnp.sum(gamma, axis=1)
            new_weights = n_k / jnp.sum(n_k)
            new_means = jnp.array([
                jnp.sum(gamma[k, :, None] * samples, axis=0) / jnp.maximum(n_k[k], 1e-8)
                for k in range(n_components)
            ])
            new_covs = jnp.array([
                (gamma[k, :, None] * (samples - new_means[k])).T @ (samples - new_means[k]) / jnp.maximum(n_k[k], 1e-8)
                + 1e-6 * jnp.eye(action_dim)
                for k in range(n_components)
            ])
            ess = 1.0 / jnp.sum(sample_weights ** 2)
            return (new_means, new_covs, new_weights, rng), ess

        (means, covs, weights, rng), ess = jax.lax.scan(_em_step, (means, covs, weights, rng), None, length=n_steps)
        return means, covs, weights, ess

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Distillation loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
        actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        # Total loss.
        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        # Additional metrics for logging.
        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the one-step policy."""
        action_seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actions = self.network.select('actor_onestep_flow')(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
        )
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fql',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-5,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            flow_steps=100,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
