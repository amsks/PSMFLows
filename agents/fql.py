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

        CAN RETURN NaN. The inverse is an implicit-Euler fixed point run for a FIXED 5
        sweeps with no convergence check, and for some (state, action) it diverges instead
        of contracting. Measured on the cube Stage-A ckpt at n_steps=100: 13 of 1M dataset
        transitions (0.0013%), deterministic across rng seeds since nothing here is random.
        NaN then propagates to the Jacobian, hence to the Laplace proposal built from it in
        `_get_predistribution_proposal`, hence to every sample of the EM that starts there --
        so a NaN mixture in the preimage npz means THIS diverged, not that the EM misbehaved.
        Callers must treat non-finite output as "inversion failed for this row"; see
        `utils.flow_inversion.compute_preimage_validity`. Raising n_steps reduces but does
        not eliminate it.
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

    def _get_predistribution_proposal(self, state, action, n_steps, alpha=1.0, prior_scale=1.0):
        """Local Gaussian proposal (mean, cov) for the preimage of `action`.

        cov is the Laplace covariance of the target
        pi(x) ~ N(x; 0, I)^prior_scale * exp(-alpha*||flow(x)-action||^2) around the
        preimage: (2*alpha J^T J + prior_scale*I)^{-1}. Larger alpha (lower temperature)
        => tighter proposal.

        The 2*alpha (not alpha^2) is the second derivative of the SQUARED-norm target:
        exp(-alpha ||J d||^2) = exp(-1/2 d^T (2 alpha J^T J) d). The code carried
        alpha^2 against an un-squared norm, a pairing that matches neither target; at the
        production alpha=20 it made the proposal 10x too narrow in well-conditioned
        directions. Every alpha tuned against the old un-squared target, and every mixture
        npz written under it, is invalidated by this change.

        The prior term is what bounds the proposal from above: without it the covariance
        is (1/(2*alpha))(J^T J)^{-1}, which diverges along the directions the flow map
        flattens (small singular values of J), and the old code capped it by clipping the
        eigenvalues at 1.0 instead. Clipping puts the proposal at exactly the prior's width
        in those directions while the target it is proposing FOR is the posterior, which is
        narrower there -- a proposal/target mismatch that shows up directly as low ESS.
        `prior_scale=0.0` recovers the clipped likelihood-only proposal exactly.
        """
        x_0, jacobian = self._get_preimage_and_jacobian(state, action, n_steps)
        gram = jacobian.T @ jacobian + 1e-6 * jnp.eye(jacobian.shape[-1], dtype=jacobian.dtype)
        eigvals, eigvecs = jnp.linalg.eigh(gram)
        # positional bounds: jax removed the a_min/a_max kwargs (broke on jax 0.10). The
        # bounds stay as guards; at prior_scale=1 the upper one is already implied, since
        # (2*alpha lambda + 1)^{-1} <= 1 for every eigenvalue.
        cov_eigvals = jnp.clip(1.0 / (2.0 * alpha * eigvals + prior_scale), 0.01, 1.0)
        cov = (eigvecs * cov_eigvals[None, :]) @ eigvecs.T
        return x_0, cov
    
    def compute_full_proposal_distribution(self, state, action, rng, num_samples=100, n_steps=10, n_initial_steps=100, alpha=1.0, prior_scale=1.0, skills=None):
        """Refine the preimage proposal toward the latent POSTERIOR by importance sampling.

        pi(x) ~ N(x; 0, I)^prior_scale * exp(-alpha * ||flow(x) - action||).

        alpha is an inverse temperature (1/T): larger alpha => sharper target.
        prior_scale weights the flow's own latent prior; see the EM variant's docstring
        for why omitting it (prior_scale=0, the pre-2026-08-14 behaviour) is wrong.
        `state` is the RAW observation; with skill_cond the hindsight target `skills`
        must be passed and is threaded to every flow call (never pre-concatenate).
        """
        if self.config['skill_cond']:
            assert skills is not None, 'skill_cond=True: pass the raw state plus skills'
        x_0, cov = self._get_predistribution_proposal(
            self._actor_obs(state, skills), action, n_initial_steps, alpha, prior_scale)
        state_b = jnp.broadcast_to(state, (num_samples, *state.shape))
        skills_b = None if skills is None else jnp.broadcast_to(skills, (num_samples, *skills.shape))

        def _step(carry, _):
            x_0, cov, rng = carry
            prop_dist = distrax.MultivariateNormalFullCovariance(loc=x_0, covariance_matrix=cov)
            rng, sample_rng = jax.random.split(rng)
            samples, log_prob = prop_dist.sample_and_log_prob(seed=sample_rng, sample_shape=(num_samples,))
            actions = self.compute_flow_actions(state_b, noises=samples, skills=skills_b)
            dist = alpha * jnp.sum((actions - action[None]) ** 2, axis=-1)
            log_prior = -0.5 * prior_scale * jnp.sum(samples ** 2, axis=-1)
            # Same exposure as the EM variant: the flow diverges to NaN when integrated
            # from a tail sample at flow_steps>=100, and one NaN makes the whole softmax
            # NaN. Give such samples weight zero; uniform if none survive.
            logits = log_prior - dist - log_prob
            ok = jnp.isfinite(logits)
            logits = jnp.where(ok, logits, -jnp.inf)
            logits = jnp.where(jnp.any(ok), logits, jnp.zeros_like(logits))
            weights = jax.nn.softmax(logits, axis=0)
            # Report 0, not num_samples, when NOTHING was usable. The uniform fallback above
            # makes every weight 1/num_samples, so 1/sum(w^2) evaluates to num_samples --
            # the metric's BEST attainable value -- on total failure. See the EM variant.
            ess = jnp.where(jnp.any(ok), 1.0 / jnp.sum(weights ** 2), 0.0)
            new_x_0 = jnp.sum(weights[..., None] * samples, axis=0)
            diff = samples - new_x_0[None, :]
            new_cov = (weights[..., None] * diff).T @ diff + 1e-6 * jnp.eye(cov.shape[-1], dtype=cov.dtype)
            return (new_x_0, new_cov, rng), ess

        (x_0, cov, rng), ess = jax.lax.scan(_step, (x_0, cov, rng), None, length=n_steps)
        return x_0, cov, ess

    def compute_full_proposal_distribution_em(self, state, action, rng, num_samples=100, n_steps=10, n_initial_steps=100, alpha=1.0, n_components=3, prior_scale=1.0, skills=None):
        """Importance-weighted EM fit of a Gaussian mixture to the latent POSTERIOR

            pi(x) ~ N(x; 0, I)^prior_scale * exp(-alpha * ||flow(x) - action||).

        The prior factor is not optional decoration: the flow is a map from N(0, I) to
        actions, so the quantity a latent-space policy needs is p(u | s, a), and the
        likelihood term alone is FLAT in every direction the decoder is insensitive to.
        Fitting a Gaussian to a flat ridge has no finite answer, and the EM duly runs away
        --- measured on this code before the fix: max covariance eigenvalue 1 -> 6 -> 34 ->
        305 -> 2281 -> 6095 over 8 steps, and in the SHIPPED npz files 84% of cube rows
        (50% of pointmaze rows) carry a fitted per-dim variance wider than the prior they
        are supposed to live in, up to 9.8 on cube and 3.8e4 on pointmaze, with component
        means out at |mu| = 206. The exact backward-ODE preimages in the same files sit
        where the prior says they should (|u| mean 0.85, p99 2.7), which is the control
        that identifies the target, not the inverter, as the culprit.

        prior_scale=0.0 reproduces the old likelihood-only target exactly, for
        regenerating a legacy npz; anything else is the posterior with that prior weight.

        Samples are drawn from the current mixture; per-sample IS weights w_n ~ pi/q
        carry the energy, membership responsibilities r_{k,n} assign them to components,
        and the M-step uses gamma_{k,n} = w_n * r_{k,n}. alpha is an inverse temperature.
        `state` is the RAW observation; with skill_cond pass `skills` (as in the IS
        variant above), never a pre-concatenated state.

        `prior_scale=0.0` drops the prior, which is what this fitted before 2026-08-14 and
        what every published npz was computed under. Without it the target is a likelihood
        only: it has no term pulling mass toward the typical set, so the fit follows the
        level set of the decode error out of the region the flow was trained on. Measured
        on the published npz files, mean ||u||^2 of a mixture draw was 14.6 against
        E[chi^2_5] = 5 on cube and 59.3 against 2 on pointmaze, with 34%/43% of draws past
        the chi^2 99th percentile -- while the point preimage stayed typical (5.57 / 1.64).
        Latents that far out are pure extrapolation for the flow, and ~0.3% of them made
        the forward integration diverge outright.
        """
        if self.config['skill_cond']:
            assert skills is not None, 'skill_cond=True: pass the raw state plus skills'
        x_0, cov = self._get_predistribution_proposal(
            self._actor_obs(state, skills), action, n_initial_steps, alpha, prior_scale)
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
            skills_b = None if skills is None else jnp.broadcast_to(skills, (samples.shape[0], *skills.shape))
            actions = self.compute_flow_actions(state_b, noises=samples, skills=skills_b)
            # log pi = log prior + log likelihood, up to a constant: the -||x||^2/2 term is
            # the flow's N(0, I) latent prior (agents/fql.py actor_loss draws x_0 from it).
            log_energy = (-alpha * jnp.sum((actions - action[None]) ** 2, axis=-1)
                          - 0.5 * prior_scale * jnp.sum(samples ** 2, axis=-1))

            log_likelihoods = jax.vmap(
                lambda m, c: jax.vmap(
                    lambda x: distrax.MultivariateNormalFullCovariance(loc=m, covariance_matrix=c).log_prob(x)
                )(samples)
            )(means, covs)

            # Floor the mixture weights: a component that is driven to exactly zero makes
            # log(weights) = -inf, and then log_joint - log_q is -inf - (-inf) = NaN.
            safe_weights = jnp.maximum(weights, 1e-12)
            log_joint = jnp.log(safe_weights[..., None]) + log_likelihoods
            log_q = jax.scipy.special.logsumexp(log_joint, axis=0)
            # Drop samples we cannot score, for either of two reasons:
            #   log_q  -- a sample far from EVERY component has log_q = -inf, which poisons
            #             both the responsibilities and the importance weights.
            #   energy -- the BC flow itself DIVERGES when integrated from a proposal noise
            #             in the tail. The Laplace covariance saturates at the 1.0 clip in
            #             _get_predistribution_proposal, so samples are effectively drawn
            #             ~N(x_0, I); measured on the cube Stage-A checkpoint, ~0.02% land
            #             at ||u||~6.8 (mean 3.1) and their trajectory runs 6.5 -> 2.3e10
            #             -> inf -> NaN. flow_steps 10 and 30 under-resolve the blow-up and
            #             stay finite, which is why this only appeared at >=100.
            # Masking matters because softmax over a vector containing a single NaN is NaN
            # in EVERY position: one diverged sample took out its whole row (measured
            # cascade: 17% of rows at EM step 0, 69% at step 1, 100% by step 5). A diverged
            # sample is just a bad preimage candidate, so it takes weight zero.
            q_ok = jnp.isfinite(log_q)
            log_q = jnp.where(q_ok, log_q, 0.0)
            ok = q_ok & jnp.isfinite(log_energy)
            responsibilities = jnp.where(
                ok[None, :], jnp.exp(log_joint - log_q[None, :]), 1.0 / n_components)
            logits = jnp.where(ok, log_energy - log_q, -jnp.inf)
            # If NOTHING is usable, softmax over all -inf is NaN; fall back to uniform.
            logits = jnp.where(jnp.any(ok), logits, jnp.zeros_like(logits))
            sample_weights = jax.nn.softmax(logits, axis=0)
            # Masked samples get sample_weights exactly 0, and their responsibilities are
            # finite by construction above, so gamma stays finite (0 * NaN would not).
            gamma = responsibilities * sample_weights[None, :]

            n_k = jnp.sum(gamma, axis=1)
            new_weights = n_k / jnp.sum(n_k)
            # A component whose responsibility mass collapses gets its scatter divided by a
            # near-zero n_k, producing a huge or non-PSD covariance; the next iteration's
            # MultivariateNormalFullCovariance then returns NaN. Measured on
            # cube-single-play this hit ~1% of transitions (always at the last EM step),
            # which over precompute_preimages.py's ~1M transitions is ~10k corrupted
            # latents. Symmetrize and floor the spectrum to keep every component PSD.
            # The floor must be RELATIVE to the largest eigenvalue, not absolute: this EM
            # grows the covariance scale by ~4x per iteration (measured: max eigenvalue
            # 1 -> 6 -> 34 -> 305 -> 2281 -> 6095 over 8 steps), and an absolute 1e-6 floor
            # against a 6e3 top eigenvalue is a condition number of 1e9, which float32
            # cannot reconstruct as PSD — the eigendecomposition returns small NEGATIVE
            # eigenvalues and the next log_prob is NaN. Capping the condition number at
            # 1e6 keeps every component representable.
            def _condition(c):
                c = 0.5 * (c + jnp.swapaxes(c, -1, -2))          # kill asymmetry drift
                w, v = jnp.linalg.eigh(c)
                floor = jnp.maximum(jnp.max(w, axis=-1, keepdims=True) * 1e-6, 1e-12)
                w = jnp.maximum(w, floor)
                return (v * w[..., None, :]) @ jnp.swapaxes(v, -1, -2)
            new_means = jnp.array([
                jnp.sum(gamma[k, :, None] * samples, axis=0) / jnp.maximum(n_k[k], 1e-8)
                for k in range(n_components)
            ])
            new_covs = jnp.array([
                _condition(
                    (gamma[k, :, None] * (samples - new_means[k])).T @ (samples - new_means[k])
                    / jnp.maximum(n_k[k], 1e-8)
                    + 1e-6 * jnp.eye(action_dim)
                )
                for k in range(n_components)
            ])
            # ESS must not report success when the step failed outright. If `ok` is all-False
            # the fallback at the `logits` line above makes every weight exactly
            # 1/num_samples, so 1/sum(w^2) == num_samples -- the BEST value the metric can
            # take -- for a row where every single sample was rejected. Measured on the cube
            # Stage-A ckpt: all 13 of the 1M rows whose stored mixture is NaN reported
            # ESS=100/100, deterministically across rng seeds. D3 gates on mean ESS, so the
            # gate was blind to precisely its worst cases. 0 is the honest floor: a row with
            # no usable sample has no effective samples.
            ess = jnp.where(jnp.any(ok), 1.0 / jnp.sum(sample_weights ** 2), 0.0)
            return (new_means, new_covs, new_weights, rng), ess

        (means, covs, weights, rng), ess = jax.lax.scan(_em_step, (means, covs, weights, rng), None, length=n_steps)
        return means, covs, weights, ess

    def _actor_obs(self, observations, skills):
        """Condition actor-network inputs on the hindsight skill target.

        No-op (returns `observations` unchanged) when skill_cond is off. When on, `skills`
        must be an array shaped like `observations`; concatenation is the ONLY mechanism
        (GCBC-style), never a latent sample. `skill_cond` is a static config value, so the
        branch resolves at trace time and stays jit-compatible.
        """
        if not self.config['skill_cond']:
            return observations
        return jnp.concatenate([observations, skills], axis=-1)

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

        actor_obs = self._actor_obs(batch['observations'], batch.get('skills'))
        pred = self.network.select('actor_bc_flow')(actor_obs, x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Distillation loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises, skills=batch.get('skills'))
        actor_actions = self.network.select('actor_onestep_flow')(actor_obs, noises, params=grad_params)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        if self.config['bc_only']:
            # Reward-free behaviour-flow pretraining (PSMFlows): no critic, no Q term, and
            # crucially no read of batch['rewards']/['masks'] — the pretraining dataset has
            # neither. What survives is exactly the flow objective plus its one-step
            # distillation, which is the G_theta that psmflow later freezes.
            actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss
            return actor_loss, {
                'actor_loss': actor_loss,
                'bc_flow_loss': bc_flow_loss,
                'distill_loss': distill_loss,
            }

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

        # bc_only: skip the critic branch entirely. Its params stay in the tree (so
        # checkpoints round-trip into a default-shaped agent) but receive no gradient.
        critic_loss = 0.0
        if not self.config['bc_only']:
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
        skills=None,
    ):
        """Sample actions from the one-step policy.

        `skills` is required (raises otherwise) when skill_cond is on; unused when off.
        """
        if self.config['skill_cond'] and skills is None:
            raise ValueError('skill_cond=True requires `skills` in sample_actions().')
        action_seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actor_obs = self._actor_obs(observations, skills)
        actions = self.network.select('actor_onestep_flow')(actor_obs, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
        skills=None,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        observations = self._actor_obs(observations, skills)
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

        # skill_cond: actor networks take concat([observations, skills], -1). Skills live
        # in observation space, so ex_observations doubles as the init-time skills example
        # -- only its shape matters here, not its content.
        actor_ex_obs = ex_observations
        if config['skill_cond']:
            actor_ex_obs = jnp.concatenate([ex_observations, ex_observations], axis=-1)

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
            actor_bc_flow=(actor_bc_flow_def, (actor_ex_obs, ex_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (actor_ex_obs, ex_actions)),
        )
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (actor_ex_obs,))
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
            bc_only=False,  # Reward-free behaviour-flow pretraining: drop the critic and the
            # Q term, keeping bc_flow_loss + alpha*distill_loss. The param tree is unchanged
            # (critic present but untrained) so checkpoints restore into a default FQL agent.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            skill_cond=False,  # True => actor flow is G(s, c, u): concat the hindsight-window
            # skill target c (batch['skills']) onto observations for actor_bc_flow and
            # actor_onestep_flow (GCBC-style, not a latent-variable skill VAE). False is a
            # byte-identical no-op; the critic path is unaffected either way.
            skill_window=100,  # Steps ahead for the hindsight skill target, clipped at episode end.
        )
    )
    return config
