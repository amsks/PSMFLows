"""PSM (Proto Successor Measure) agent — JAX/Flax port of the PyTorch reference.

Code <-> paper (arXiv 2411.19418):
  phi (PhiMap)          -> phi_s(s+)     basis over future states (the proto basis)
  sf_psi (PsiMap)       -> psi^pi(s,a)   successor features of the task (continuous-z) policy
  proto_psi (PsiMap)    -> psi^{pi_z}    successor features of the codebook policies pi_z
  task_z / StepInputs   -> w             task coordinates (inferred from reward at eval)
  proto_seed            -> z             binary codebook seed in pi(a|s,z)
  M = psi . phi^T       -> M^pi(s,a,s+)  the successor measure
  proto_next_action     -> pi(a|s,z)=UniformSample(z+hash(s))  codebook behavior policy
  contrastive_loss      -> Eq.7 off-diag (TD residual) + diag (source) terms
  ortho_loss            -> orthonormality regularizer (phi phi^T -> I)
  infer_z: w = E[r.phi] -> reward inference (closed form)

The update is a 3-stage SEQUENTIAL procedure (proto -> sf -> actor): each network is
its own `TrainState` (module + optimizer), stepped in turn with target soft-updates
interleaved; the SF stage reads the phi the proto stage just updated. This differs from
FQL's single-optimizer combined loss, so PSM uses one TrainState per network rather than
a single shared ModuleDict optimizer.
"""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from utils.flax_utils import TrainState, nonpytree_field
from utils.psm_networks import (
    FlowVectorField, NoiseConditionedActor, PhiMap, PsiMap, PSMActor,
    truncated_clamp, truncated_sample,
)


@flax.struct.dataclass
class StepInputs:
    """The per-step sampled quantities (replaces the old `inj` dict).

    task_z:            continuous task vector w (Gaussian, 50%-mixed with phi(goal))
    proto_seed:        binary codebook seed z for the proto branch
    proto_next_action: codebook policy action at s' (proto measure target)
    sf_next_action:    learned-actor action at s' (sf measure target)
    actor_sample:      truncated-normal actor sample (ddpgbc only; None for flow)
    flow_x0/flow_t/flow_noise: flow-BC noise/time/noise (flow only; None for ddpgbc)
    """
    task_z: Any
    proto_seed: Any
    proto_next_action: Any
    sf_next_action: Any
    actor_sample: Any = None
    flow_x0: Any = None
    flow_t: Any = None
    flow_noise: Any = None


# ----------------------------- pure helpers -----------------------------

def contrastive_loss(M, target_M, discount, off_diag, off_diag_sum):
    diff = M - discount * target_M
    offdiag = 0.5 * jnp.sum((diff * off_diag) ** 2) / off_diag_sum
    diag = -jnp.mean(jnp.diagonal(diff, axis1=1, axis2=2)) * M.shape[0]
    return offdiag + diag, diag, offdiag


def ortho_loss(phi, off_diag, off_diag_sum):
    cov = phi @ phi.T
    offdiag = 0.5 * jnp.sum((cov * off_diag) ** 2) / off_diag_sum
    diag = -jnp.mean(jnp.diagonal(cov))
    return offdiag + diag, diag, offdiag


def targets_uncertainty(preds, num_parallel):
    mean = preds.mean(axis=0)
    d1 = preds[None]
    d2 = preds[:, None]
    unc = jnp.sum(jnp.abs(d1 - d2), axis=(0, 1)) / (num_parallel ** 2 - num_parallel)
    return mean, unc


def proto_sample(seed_to_action, powers, obs_hash, z, max_seed):
    seed_long = jnp.sum(z * powers, axis=1)
    final = ((seed_long + obs_hash.reshape(-1)) % max_seed).astype(jnp.int32)
    return seed_to_action[final].astype(jnp.float32)


def project_z(z, norm_z):
    if not norm_z:
        return z
    d = z.shape[-1]
    return jnp.sqrt(d) * z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)


def off_diagonal_mask(B):
    """(1 - I, its sum): selects the s+ != s (off-diagonal) entries of the BxB grid."""
    off = 1.0 - jnp.eye(B)
    return off, off.sum()


def polyak_update(online, target, tau):
    """Soft-update a target param tree toward `online` at rate `tau`."""
    return jax.tree_util.tree_map(lambda p, tp: p * tau + tp * (1 - tau), online, target)


# ----------------------------- agent -----------------------------

class PSMAgent(flax.struct.PyTreeNode):
    """PSM agent: one TrainState per network + plain target param trees.

    Networks: phi (basis), proto_psi (codebook head), sf_psi (task head), actor
    (+ actor_vf for the flow actor). Targets are held as plain param pytrees and
    soft-updated during `apply_update`.
    """

    rng: Any
    phi: TrainState
    proto_psi: TrainState
    sf_psi: TrainState
    actor: TrainState
    target_phi: Any
    target_proto_psi: Any
    target_sf_psi: Any
    task_z: Any            # (z_dim,) eval task vector w, set via infer_eval_z
    config: Any = nonpytree_field()
    actor_vf: Any = None   # TrainState for the flow actor's velocity field; None for ddpgbc
    proto: Any = None      # (seed_to_action, powers) — traced pytree; max_seed in config

    # ---- branch losses ----
    def proto_loss(self, batch, sampled, phi_params, proto_psi_params):
        """Proto branch: learn the basis phi from the codebook policies' measure."""
        c = self.config
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        goal = next_obs  # phi_input='s', Identity normalizer/encoder for state
        off, off_sum = off_diagonal_mask(obs.shape[0])
        P = c["num_parallel"]
        z_psm, proto_na = sampled.proto_seed, sampled.proto_next_action

        phi_g = self.phi(goal, params=phi_params)
        M = self.proto_psi(obs, z_psm, action, params=proto_psi_params) @ phi_g.T
        tphi = self.phi(goal, params=self.target_phi)
        tM = self.proto_psi(next_obs, z_psm, proto_na, params=self.target_proto_psi) @ tphi.T
        tmean, tunc = targets_uncertainty(tM, P)
        target_M = tmean - c["pessimism_penalty"] * tunc
        cl, cdiag, coff = contrastive_loss(M, jax.lax.stop_gradient(target_M), c["discount"], off, off_sum)
        ol, odiag, ooff = ortho_loss(phi_g, off, off_sum)
        loss = cl + c["ortho_coef"] * ol
        return loss, {"psm_loss": loss, "psm_diag": cdiag, "psm_offdiag": coff,
                      "orth_loss": ol, "orth_diag": odiag, "orth_offdiag": ooff}

    def sf_loss(self, batch, sampled, sf_params, phi_params):
        """SF branch: fit the task successor features on the (frozen) basis phi.

        `phi_params` is the *just-updated* online phi; the target measure uses the same
        online phi (NOT target_phi, matching the reference) and target_sf_psi.
        """
        c = self.config
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        goal = next_obs
        off, off_sum = off_diagonal_mask(obs.shape[0])
        P = c["num_parallel"]
        z, sf_na = sampled.task_z, sampled.sf_next_action

        phi_g = jax.lax.stop_gradient(self.phi(goal, params=phi_params))  # phi frozen for SF
        M = self.sf_psi(obs, z, action, params=sf_params) @ phi_g.T
        tphi = self.phi(goal, params=phi_params)  # online phi, NOT target_phi (reference)
        tM = self.sf_psi(next_obs, z, sf_na, params=self.target_sf_psi) @ tphi.T
        tmean, tunc = targets_uncertainty(tM, P)
        target_M = tmean - c["pessimism_penalty"] * tunc
        cl, cdiag, coff = contrastive_loss(M, jax.lax.stop_gradient(target_M), c["discount"], off, off_sum)
        return cl, {"sf_loss": cl, "sf_diag": cdiag, "sf_offdiag": coff}

    def actor_loss(self, batch, sampled, actor_params, sf_params):
        """DDPGBC actor: TD3 mean + truncated exploration sample, optional BC term.

        `sf_params` (the just-updated sf_psi) supplies Q; it is a constant here.
        """
        c = self.config
        obs, action = batch["observations"], batch["actions"]
        z = sampled.task_z
        P = c["num_parallel"]
        mu = self.actor(obs, z, params=actor_params)
        a = truncated_clamp(mu + jax.lax.stop_gradient(sampled.actor_sample - mu))
        qpsis = self.sf_psi(obs, z, a, params=sf_params)
        Qs = (qpsis * z).sum(-1)
        qmean, qunc = targets_uncertainty(Qs, P)
        Q = qmean - c["actor_pessimism_penalty"] * qunc
        loss = -Q.mean()
        info = {"actor_loss": loss, "q": Q.mean()}
        if c["bc_coeff"] > 0:
            bc_error = jnp.mean((a - action) ** 2)
            # normalize Q by |Q| so the BC term has a stable relative scale (td_jepa/FB).
            loss = loss / jax.lax.stop_gradient(jnp.abs(Qs).mean()) + c["bc_coeff"] * bc_error
            info = {"actor_loss": loss, "q": Q.mean(), "bc_error": bc_error}
        return loss, info

    def flow_actor_loss(self, batch, sampled, actor_params, vf_params, sf_params):
        """Flow (FQL-style) actor: BC flow-matching velocity field v(s, x_t, t) plus a
        one-step, z-conditioned noise actor distilled from its ODE rollout. Q comes from
        the (fixed) sf_psi. Differentiated w.r.t. (actor_params, vf_params)."""
        c = self.config
        obs, action = batch["observations"], batch["actions"]
        z = sampled.task_z
        x0, t, noise = sampled.flow_x0, sampled.flow_t, sampled.flow_noise
        P = c["num_parallel"]
        bc_coeff = c["bc_coeff"]
        steps = c["flow_steps"]

        def rollout(vf_p, o, n):
            a = n
            for i in range(steps):
                ti = jnp.full((o.shape[0], 1), i / steps)
                a = a + self.actor_vf(o, a, ti, params=vf_p) / steps
            return jnp.clip(a, -1.0, 1.0)

        # BC flow-matching loss (trains the velocity field on dataset actions).
        x1 = action
        xt = (1 - t) * x0 + t * x1
        vel = x1 - x0
        pred = self.actor_vf(obs, xt, t, params=vf_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)
        # One-step z-conditioned actor action (NoiseConditionedActor applies tanh internally).
        a = self.actor(obs, z, noise, params=actor_params)
        qpsis = self.sf_psi(obs, z, a, params=sf_params)
        Qs = (qpsis * z).sum(-1)
        qmean, qunc = targets_uncertainty(Qs, P)
        Q = qmean - c["actor_pessimism_penalty"] * qunc
        q_loss = -Q.mean()
        info = {"q": Q.mean(), "bc_flow_loss": bc_flow_loss}
        if bc_coeff > 0:
            target = jax.lax.stop_gradient(rollout(vf_params, obs, noise))
            distill = jnp.mean((a - target) ** 2)
            q_loss = q_loss / jax.lax.stop_gradient(jnp.abs(Qs).mean()) + bc_coeff * distill
            info["bc_error"] = distill
        loss = q_loss + bc_flow_loss
        info["actor_loss"] = loss
        return loss, info

    # ---- update orchestration ----
    def losses_and_grads(self, batch, sampled):
        """Compute the three branch losses + grads at the CURRENT params, with NO
        interleaving (matches the fixture's no-step static export)."""
        (_, psm_i), (g_phi, g_proto) = jax.value_and_grad(self.proto_loss, argnums=(2, 3), has_aux=True)(
            batch, sampled, self.phi.params, self.proto_psi.params)
        (_, sf_i), g_sf = jax.value_and_grad(self.sf_loss, argnums=2, has_aux=True)(
            batch, sampled, self.sf_psi.params, self.phi.params)
        if self.config["actor_type"] == "flow":
            (_, a_i), (g_actor, g_vf) = jax.value_and_grad(self.flow_actor_loss, argnums=(2, 3), has_aux=True)(
                batch, sampled, self.actor.params, self.actor_vf.params, self.sf_psi.params)
            grads = {"phi": g_phi, "proto_psi": g_proto, "sf_psi": g_sf, "actor": g_actor, "actor_vf": g_vf}
        else:
            (_, a_i), g_actor = jax.value_and_grad(self.actor_loss, argnums=2, has_aux=True)(
                batch, sampled, self.actor.params, self.sf_psi.params)
            grads = {"phi": g_phi, "proto_psi": g_proto, "sf_psi": g_sf, "actor": g_actor}
        return {**psm_i, **sf_i, **a_i}, grads

    def apply_update(self, batch, sampled):
        """3-stage interleaved update (matches the fixture K-step trace)."""
        tau = self.config["tau"]

        # stage 1: proto -> step phi + proto_psi, then soft-update their targets
        (_, psm_i), (g_phi, g_proto) = jax.value_and_grad(self.proto_loss, argnums=(2, 3), has_aux=True)(
            batch, sampled, self.phi.params, self.proto_psi.params)
        phi = self.phi.apply_gradients(grads=g_phi)
        proto_psi = self.proto_psi.apply_gradients(grads=g_proto)
        target_phi = polyak_update(phi.params, self.target_phi, tau)
        target_proto_psi = polyak_update(proto_psi.params, self.target_proto_psi, tau)

        # stage 2: sf reads the just-updated phi -> step sf_psi, soft-update its target.
        # sf_loss reads self.target_sf_psi (still the pre-update target) + phi.params (updated).
        (_, sf_i), g_sf = jax.value_and_grad(self.sf_loss, argnums=2, has_aux=True)(
            batch, sampled, self.sf_psi.params, phi.params)
        sf_psi = self.sf_psi.apply_gradients(grads=g_sf)
        target_sf_psi = polyak_update(sf_psi.params, self.target_sf_psi, tau)

        # stage 3: actor reads the just-updated sf_psi
        actor_vf = self.actor_vf
        if self.config["actor_type"] == "flow":
            (_, a_i), (g_actor, g_vf) = jax.value_and_grad(self.flow_actor_loss, argnums=(2, 3), has_aux=True)(
                batch, sampled, self.actor.params, self.actor_vf.params, sf_psi.params)
            actor = self.actor.apply_gradients(grads=g_actor)
            actor_vf = self.actor_vf.apply_gradients(grads=g_vf)
        else:
            (_, a_i), g_actor = jax.value_and_grad(self.actor_loss, argnums=2, has_aux=True)(
                batch, sampled, self.actor.params, sf_psi.params)
            actor = self.actor.apply_gradients(grads=g_actor)

        info = {**psm_i, **sf_i, **a_i}
        return self.replace(phi=phi, proto_psi=proto_psi, sf_psi=sf_psi, actor=actor, actor_vf=actor_vf,
                            target_phi=target_phi, target_proto_psi=target_proto_psi,
                            target_sf_psi=target_sf_psi), info

    def sample_step_inputs(self, batch, rng):
        """Sample the per-step quantities (task vector, codebook seed, next actions,
        flow noise) as a named `StepInputs`."""
        c = self.config
        B = batch["observations"].shape[0]
        adim = c["action_dim"]
        obs, next_obs = batch["observations"], batch["next_observations"]
        r1, r2, r5, rperm, rtail = jax.random.split(rng, 5)
        # task vector w: Gaussian, with a mix_ratio fraction replaced by project_z(phi(goal[perm]))
        # (reference sample_mixed_z, as a jit-friendly mask instead of dynamic indexing).
        gauss_z = project_z(jax.random.normal(r1, (B, c["z_dim"])), c["norm_z"])
        goal = next_obs
        perm = jax.random.permutation(rperm, B)
        mixed_z = project_z(self.phi(goal)[perm], c["norm_z"])
        mix_mask = (jax.random.uniform(r5, (B,)) < c["mix_ratio"])[:, None]
        task_z = jnp.where(mix_mask, mixed_z, gauss_z)
        # binary codebook seed z for the proto branch.
        zbin = (jax.random.randint(r2, (B,), 0, 2 ** c["max_log_seed"])[:, None]
                & (1 << jnp.arange(c["max_log_seed"]))) > 0
        proto_seed = zbin.astype(jnp.float32)
        seed_to_action, powers = self.proto
        max_seed = c["proto_max_seed"]
        # Proto behavior policy is keyed on the GLOBAL replay-buffer row index of each
        # transition (reference next_obs_hash = batch["index"]). Falling back to arange(B)
        # (batch position) makes the proto next-action depend on where a transition lands in
        # the batch — inconsistent across resamples. Use the injected index when present.
        obs_hash = batch["index"] if "index" in batch else jnp.arange(B)
        proto_next_action = proto_sample(seed_to_action, powers, obs_hash, proto_seed, max_seed)

        if c["actor_type"] == "flow":
            r_x0, r_t, r_noise, r_na = jax.random.split(rtail, 4)
            # SF-branch next action = the one-step flow policy at s' (bootstrap target).
            na = self.actor(next_obs, task_z, jax.random.normal(r_na, (B, adim)))
            return StepInputs(
                task_z=task_z, proto_seed=proto_seed, proto_next_action=proto_next_action,
                sf_next_action=jnp.clip(na, -1.0, 1.0), actor_sample=None,
                flow_x0=jax.random.normal(r_x0, (B, adim)),
                flow_t=jax.random.uniform(r_t, (B, 1)),
                flow_noise=jax.random.normal(r_noise, (B, adim)))
        r3, r4 = jax.random.split(rtail)
        mu_next = self.actor(next_obs, task_z)
        sf_next_action = truncated_sample(mu_next, c["actor_std"], jax.random.normal(r3, (B, adim)), clip=c["stddev_clip"])
        mu = self.actor(obs, task_z)
        actor_sample = truncated_sample(mu, c["actor_std"], jax.random.normal(r4, (B, adim)), clip=c["stddev_clip"])
        return StepInputs(
            task_z=task_z, proto_seed=proto_seed, proto_next_action=proto_next_action,
            sf_next_action=sf_next_action, actor_sample=actor_sample,
            flow_x0=None, flow_t=None, flow_noise=None)

    @jax.jit
    def update(self, batch):
        # Jitted training step. Each TrainState carries its (nonpytree) module/optimizer as
        # static aux and its params/opt_state as traced leaves, so `self` is a valid jit
        # argument. The equivalence tests call apply_update/losses_and_grads directly
        # (un-jitted), so their numerics are unaffected by any op-fusion here.
        new_rng, rng = jax.random.split(self.rng)
        sampled = self.sample_step_inputs(batch, rng)
        new_agent, info = self.apply_update(batch, sampled)
        return new_agent.replace(rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        # PSM acts conditioned on the task latent task_z (inferred from rewards via
        # infer_eval_z; defaults to zeros until inferred). DDPGBC returns the TD3 actor
        # mean; flow runs the one-step, z-conditioned noise actor.
        z = jnp.broadcast_to(self.task_z, (*observations.shape[:-1], self.config["z_dim"]))
        if self.config["actor_type"] == "flow":
            seed = self.rng if seed is None else seed
            noise = jax.random.normal(seed, (*observations.shape[:-1], self.config["action_dim"]))
            a = self.actor(observations, z, noise)
            return jnp.clip(a, -1.0, 1.0)
        return self.actor(observations, z)

    def total_loss(self, batch, grad_params=None, rng=None):
        """Validation-logging loss: the three branch losses at current params (no step)."""
        rng = rng if rng is not None else self.rng
        sampled = self.sample_step_inputs(batch, rng)
        psm_l, psm_i = self.proto_loss(batch, sampled, self.phi.params, self.proto_psi.params)
        sf_l, sf_i = self.sf_loss(batch, sampled, self.sf_psi.params, self.phi.params)
        if self.config["actor_type"] == "flow":
            a_l, a_i = self.flow_actor_loss(batch, sampled, self.actor.params, self.actor_vf.params, self.sf_psi.params)
        else:
            a_l, a_i = self.actor_loss(batch, sampled, self.actor.params, self.sf_psi.params)
        return psm_l + sf_l + a_l, {**psm_i, **sf_i, **a_i}

    def infer_z(self, next_observations, rewards):
        phi = self.phi(next_observations)
        z = (rewards.reshape(1, -1) @ phi).reshape(-1) / phi.shape[0]
        return project_z(z, self.config["norm_z"])

    def infer_eval_z(self, next_observations, rewards):
        """Return a copy of this agent with task_z set to the reward-inferred task
        latent. Call on the TRAINED agent (phi must be trained) before eval acting."""
        return self.replace(task_z=self.infer_z(next_observations, rewards))

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, rphi, rsf, rpsm, ract, rproto = jax.random.split(rng, 6)
        # PSM's bespoke phi/psi/actor operate on raw states; visual encoders are not
        # supported yet (pixel envs are out of scope). Fail loudly rather than silently
        # ignore a requested encoder.
        assert config.get("encoder", None) is None, "PSM does not support visual encoders yet."
        action_dim = ex_actions.shape[-1]
        z_dim = config["z_dim"]
        actor_cfg = config["actor"]
        actor_type = actor_cfg["type"] if "type" in actor_cfg else "ddpgbc"
        assert actor_type in ("ddpgbc", "flow"), f"unknown actor.type {actor_type!r}"

        ex_obs = ex_observations
        ex_z = jnp.zeros((ex_obs.shape[0], z_dim))
        ex_zbin = jnp.zeros((ex_obs.shape[0], config["max_log_seed"]))

        phi_def = PhiMap(z_dim=z_dim, hidden_dim=config["phi"]["hidden_dim"],
                         hidden_layers=config["phi"]["hidden_layers"], norm=True)
        psi_kw = dict(output_dim=z_dim, hidden_dim=config["sf"]["hidden_dim"],
                      num_parallel=config["num_parallel"],
                      embedding_layers=config["sf"]["embedding_layers"],
                      hidden_layers=config["sf"]["hidden_layers"])
        sf_def, proto_def = PsiMap(**psi_kw), PsiMap(**psi_kw)

        phi = TrainState.create(phi_def, phi_def.init(rphi, ex_obs)["params"],
                                tx=optax.adam(config["lr_phi"]))
        sf_psi = TrainState.create(sf_def, sf_def.init(rsf, ex_obs, ex_z, ex_actions)["params"],
                                   tx=optax.adam(config["lr_sf"]))
        proto_psi = TrainState.create(proto_def, proto_def.init(rpsm, ex_obs, ex_zbin, ex_actions)["params"],
                                      tx=optax.adam(config["lr_sf"]))

        lr_actor_vf = float(actor_cfg["lr_actor_vf"]) if "lr_actor_vf" in actor_cfg else 3e-4
        actor_vf = None
        if actor_type == "flow":
            # Flow actor: a faithful NoiseConditionedActor head (a = tanh(net(obs, z, noise))
            # via dual LayerNorm/Tanh/ReLU embeddings) + an unconditional GELU velocity field
            # v(s, x_t, t). Ports agents/psm/flow_bc + nn_models.{NoiseConditionedActor,VectorField}.
            def _acfg(key, default):
                return actor_cfg[key] if key in actor_cfg else default
            actor_def = NoiseConditionedActor(
                action_dim=action_dim, hidden_dim=int(_acfg("flow_actor_hidden_dim", 512)),
                hidden_layers=int(_acfg("flow_actor_hidden_layers", 2)),
                embedding_layers=int(_acfg("flow_actor_embedding_layers", 2)))
            vf_def = FlowVectorField(action_dim=action_dim, hidden_dim=int(_acfg("flow_vf_hidden_dim", 512)),
                                     hidden_layers=int(_acfg("flow_vf_hidden_layers", 4)))
            ract, rvf = jax.random.split(ract)
            ex_noise = ex_actions
            ex_times = ex_actions[..., :1]
            actor = TrainState.create(actor_def, actor_def.init(ract, ex_obs, ex_z, ex_noise)["params"],
                                      tx=optax.adam(config["lr_actor"]))
            actor_vf = TrainState.create(vf_def, vf_def.init(rvf, ex_obs, ex_actions, ex_times)["params"],
                                         tx=optax.adam(lr_actor_vf))
        else:
            actor_def = PSMActor(action_dim=action_dim, hidden_dim=actor_cfg["hidden_dim"],
                                 embedding_layers=actor_cfg["embedding_layers"],
                                 hidden_layers=actor_cfg["hidden_layers"])
            actor = TrainState.create(actor_def, actor_def.init(ract, ex_obs, ex_z)["params"],
                                      tx=optax.adam(config["lr_actor"]))

        target_phi = copy.deepcopy(phi.params)
        target_proto_psi = copy.deepcopy(proto_psi.params)
        target_sf_psi = copy.deepcopy(sf_psi.params)

        # proto table (jax-generated; used only for training — tests inject it). The table
        # + powers are a traced pytree leaf (so `update` can be jitted); max_seed is a
        # static python int kept in config.
        max_seed = 2 ** config["max_log_seed"] + 20000
        # Optional: transplant the reference torch proto table (per-row manual_seed(i))
        # to remove the last cross-framework RNG difference in the behavior policy. Same
        # distribution as the JAX draw ((rand-1)*2 in [-2,0)); only the exact draws differ.
        proto_path = config.get("proto_table_path", None)
        if proto_path:
            import numpy as _np
            _t = _np.load(proto_path)
            assert _t.shape == (max_seed, action_dim), \
                f"proto table {_t.shape} != expected {(max_seed, action_dim)}"
            table = jnp.asarray(_t, jnp.float32)
        else:
            table = (jax.random.uniform(rproto, (max_seed, action_dim)) - 1.0) * 2.0
        powers = (2 ** jnp.arange(config["max_log_seed"]))[::-1].astype(jnp.float32)
        proto = (table.astype(jnp.float32), powers)

        # Store a fully-plain config (nested ConfigDicts -> dicts, lists -> tuples) so the
        # FrozenDict is hashable and can serve as the jit static aux for `update`.
        config = _plain_config(config)
        config["ob_dims"] = tuple(ex_observations.shape[1:])
        config["action_dim"] = action_dim
        config["proto_max_seed"] = max_seed
        # Hoist actor-dispatch scalars to the top level so runtime methods read plain
        # values (equiv-test configs omit these -> ddpgbc with bc_coeff=0 == legacy path).
        config["actor_type"] = actor_type
        config["bc_coeff"] = float(actor_cfg["bc_coeff"]) if "bc_coeff" in actor_cfg else 0.0
        config["flow_steps"] = int(actor_cfg["flow_steps"]) if "flow_steps" in actor_cfg else 10
        config["lr_actor_vf"] = lr_actor_vf
        return cls(rng=rng, phi=phi, proto_psi=proto_psi, sf_psi=sf_psi, actor=actor, actor_vf=actor_vf,
                   target_phi=target_phi, target_proto_psi=target_proto_psi, target_sf_psi=target_sf_psi,
                   task_z=jnp.zeros((z_dim,), jnp.float32), config=flax.core.FrozenDict(config), proto=proto)


def get_config():
    """Importable default config, mirroring configs/agent/psm.yaml. Uses the same
    ml_collections placeholder pattern as the other agents for auto-set fields."""
    import ml_collections

    return ml_collections.ConfigDict(
        dict(
            agent_name="psm",
            batch_size=1024,
            z_dim=128,
            max_log_seed=16,
            num_parallel=2,
            discount=0.98,
            tau=0.01,
            ortho_coef=1.0,
            mix_ratio=0.5,
            pessimism_penalty=0.0,
            actor_pessimism_penalty=0.5,
            actor_std=0.2,
            stddev_clip=0.3,
            norm_z=True,
            phi_input="s",
            lr_phi=1.0e-5,  # reference PSM sweep winner (configs/agent/psm.yaml); NOT 1e-4
            lr_sf=1.0e-4,
            lr_actor=1.0e-4,
            phi=dict(hidden_dim=256, hidden_layers=2),
            sf=dict(hidden_dim=1024, hidden_layers=1, embedding_layers=2),
            actor=dict(type="ddpgbc", hidden_dim=1024, hidden_layers=1, embedding_layers=2,
                       bc_coeff=3.0, flow_steps=10, lr_actor_vf=3.0e-4,
                       # flow actor (NoiseConditionedActor) + velocity field (VectorField) dims,
                       # matching reference configs/agent/psm_flowbc.yaml.
                       flow_actor_hidden_dim=512, flow_actor_hidden_layers=2,
                       flow_actor_embedding_layers=2,
                       flow_vf_hidden_dim=512, flow_vf_hidden_layers=4),
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )


def _plain_config(x):
    """Recursively convert ConfigDict/dict -> plain dict and list -> tuple, so the
    resulting FrozenDict is hashable (required for the jitted `update`'s static aux)."""
    if isinstance(x, (list, tuple)):
        return tuple(_plain_config(v) for v in x)
    if hasattr(x, "items"):
        return {k: _plain_config(v) for k, v in x.items()}
    return x


# ---------------------------------------------------------------------------
# Shared helpers still imported by agents/fb.py (the FB agent has not yet been
# migrated to the per-network TrainState structure). PSM itself no longer uses
# these — they will move to a shared module when FB is refactored.
# ---------------------------------------------------------------------------

class _HashableDict(dict):
    """A dict that hashes/compares by identity so it can live in a jit static aux."""

    __hash__ = object.__hash__

    def __eq__(self, other):
        return self is other


def _step(tx, grad, params, opt_state):
    updates, new_opt = tx.update(grad, opt_state, params)
    return optax.apply_updates(params, updates), new_opt


def _soft(online, target, tau):
    return jax.tree_util.tree_map(lambda p, tp: p * tau + tp * (1 - tau), online, target)
