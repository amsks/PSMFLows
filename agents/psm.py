"""PSM (Proto Successor Measure) agent — JAX/Flax port of the PyTorch reference.

Math/op-order is transcribed verbatim from the reference
agents/psm/{agent,model,proto_sampler}.py. The update is a 3-stage SEQUENTIAL
procedure (proto -> sf -> actor), each stage stepping its own Adam optimizer with
target soft-updates interleaved; the SF branch reads the phi just updated by the
proto branch. This differs from FQL's single-optimizer combined loss.
"""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from utils.flax_utils import nonpytree_field
from utils.psm_networks import PhiMap, PsiMap, PSMActor, truncated_clamp, truncated_sample


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


# ----------------------------- agent -----------------------------

class PSMAgent(flax.struct.PyTreeNode):
    rng: Any
    params: Any            # dict: phi/psm_psi/sf_psi/actor + target_phi/target_psm_psi/target_sf_psi
    opt_states: Any        # dict: phi/psm_psi/sf_psi/actor
    config: Any = nonpytree_field()
    nets: Any = nonpytree_field()     # dict of nn.Module defs
    txs: Any = nonpytree_field()      # dict of optax GradientTransformation
    proto: Any = nonpytree_field()    # (seed_to_action, powers, max_seed)

    # ---- stage loss functions (closures built per-batch in _stage_losses) ----
    def _apply(self, name, params, *args):
        return self.nets[name].apply({"params": params}, *args)

    def _stages(self, params, batch, inj, off, off_sum):
        """Return three loss closures that read `params` (a dict). Each returns
        (loss, metrics). target nets are read from `params` (stop-grad implicit:
        only the differentiated subtree flows grad)."""
        c = self.config
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        goal = next_obs  # phi_input='s', Identity normalizer/encoder for state
        B = obs.shape[0]
        disc = c["discount"] * batch["masks"].reshape(-1, 1)  # masks = 1 - terminated (FQL/OGBench)
        P = c["num_parallel"]
        seed_to_action, powers, max_seed = self.proto

        z_psm = inj["z_psm"]
        proto_na = inj["proto_next_action"]

        def psm_loss_fn(phi_p, psm_p):
            phi_g = self._apply("phi", phi_p, goal)
            M = self._apply("psm_psi", psm_p, obs, z_psm, action) @ phi_g.T
            tphi = self._apply("phi", params["target_phi"], goal)
            tM = self._apply("psm_psi", params["target_psm_psi"], next_obs, z_psm, proto_na) @ tphi.T
            tmean, tunc = targets_uncertainty(tM, P)
            target_M = tmean - c["pessimism_penalty"] * tunc
            cl, cdiag, coff = contrastive_loss(M, jax.lax.stop_gradient(target_M), disc, off, off_sum)
            ol, odiag, ooff = ortho_loss(phi_g, off, off_sum)
            loss = cl + c["ortho_coef"] * ol
            return loss, {"psm_loss": loss, "psm_diag": cdiag, "psm_offdiag": coff,
                          "orth_loss": ol, "orth_diag": odiag, "orth_offdiag": ooff}

        z = inj["z_cont"]
        sf_na = inj["actor_next_action"]

        def sf_loss_fn(sf_p, phi_p):
            phi_g = jax.lax.stop_gradient(self._apply("phi", phi_p, goal))
            M = self._apply("sf_psi", sf_p, obs, z, action) @ phi_g.T
            tphi = self._apply("phi", phi_p, goal)  # self.phi, NOT target_phi
            tM = self._apply("sf_psi", params["target_sf_psi"], next_obs, z, sf_na) @ tphi.T
            tmean, tunc = targets_uncertainty(tM, P)
            target_M = tmean - c["pessimism_penalty"] * tunc
            cl, cdiag, coff = contrastive_loss(M, jax.lax.stop_gradient(target_M), disc, off, off_sum)
            return cl, {"sf_loss": cl, "sf_diag": cdiag, "sf_offdiag": coff}

        actor_smp = inj["actor_sample"]

        def actor_loss_fn(actor_p, sf_p):
            mu = self._apply("actor", actor_p, obs, z)
            a = truncated_clamp(mu + jax.lax.stop_gradient(actor_smp - mu))
            qpsis = self._apply("sf_psi", sf_p, obs, z, a)
            Qs = (qpsis * z).sum(-1)
            qmean, qunc = targets_uncertainty(Qs, P)
            Q = qmean - c["actor_pessimism_penalty"] * qunc
            loss = -Q.mean()
            return loss, {"actor_loss": loss, "q": Q.mean()}

        return psm_loss_fn, sf_loss_fn, actor_loss_fn

    def _off(self, B):
        off = 1.0 - jnp.eye(B)
        return off, off.sum()

    def compute_static(self, batch, inj):
        """Compute the three stage losses + grads at the CURRENT params, with NO
        interleaving (matches the fixture's no-step static export)."""
        B = batch["observations"].shape[0]
        off, off_sum = self._off(B)
        psm_fn, sf_fn, actor_fn = self._stages(self.params, batch, inj, off, off_sum)
        (psm_l, psm_i), (g_phi, g_psm) = jax.value_and_grad(psm_fn, argnums=(0, 1), has_aux=True)(
            self.params["phi"], self.params["psm_psi"])
        (sf_l, sf_i), g_sf = jax.value_and_grad(sf_fn, has_aux=True)(
            self.params["sf_psi"], self.params["phi"])
        (a_l, a_i), g_actor = jax.value_and_grad(actor_fn, has_aux=True)(
            self.params["actor"], self.params["sf_psi"])
        info = {**psm_i, **sf_i, **a_i}
        grads = {"phi": g_phi, "psm_psi": g_psm, "sf_psi": g_sf, "actor": g_actor}
        return info, grads

    def apply_update(self, batch, inj):
        """3-stage interleaved update (matches the fixture K-step trace)."""
        B = batch["observations"].shape[0]
        off, off_sum = self._off(B)
        params = dict(self.params)
        opt = dict(self.opt_states)
        tau = self.config["tau"]

        # stage 1: proto -> step phi + psm_psi, then soft-update their targets
        psm_fn, _, _ = self._stages(params, batch, inj, off, off_sum)
        (psm_l, psm_i), (g_phi, g_psm) = jax.value_and_grad(psm_fn, argnums=(0, 1), has_aux=True)(
            params["phi"], params["psm_psi"])
        params["phi"], opt["phi"] = _step(self.txs["phi"], g_phi, params["phi"], opt["phi"])
        params["psm_psi"], opt["psm_psi"] = _step(self.txs["psm_psi"], g_psm, params["psm_psi"], opt["psm_psi"])
        params["target_phi"] = _soft(params["phi"], params["target_phi"], tau)
        params["target_psm_psi"] = _soft(params["psm_psi"], params["target_psm_psi"], tau)

        # stage 2: sf (reads updated phi) -> step sf_psi, soft-update its target
        _, sf_fn, _ = self._stages(params, batch, inj, off, off_sum)
        (sf_l, sf_i), g_sf = jax.value_and_grad(sf_fn, has_aux=True)(params["sf_psi"], params["phi"])
        params["sf_psi"], opt["sf_psi"] = _step(self.txs["sf_psi"], g_sf, params["sf_psi"], opt["sf_psi"])
        params["target_sf_psi"] = _soft(params["sf_psi"], params["target_sf_psi"], tau)

        # stage 3: actor (reads updated sf_psi) -> step actor
        _, _, actor_fn = self._stages(params, batch, inj, off, off_sum)
        (a_l, a_i), g_actor = jax.value_and_grad(actor_fn, has_aux=True)(params["actor"], params["sf_psi"])
        params["actor"], opt["actor"] = _step(self.txs["actor"], g_actor, params["actor"], opt["actor"])

        info = {**psm_i, **sf_i, **a_i}
        return self.replace(params=params, opt_states=opt), info

    def _draw_injection(self, batch, rng):
        c = self.config
        B = batch["observations"].shape[0]
        adim = c["action_dim"]
        r1, r2, r3, r4, r5, rperm = jax.random.split(rng, 6)
        # SF-branch z: Gaussian, with a mix_ratio fraction replaced by project_z(phi(goal[perm]))
        # (reference sample_mixed_z, as a jit-friendly mask instead of dynamic indexing).
        gauss_z = project_z(jax.random.normal(r1, (B, c["z_dim"])), c["norm_z"])
        goal = batch["next_observations"]
        perm = jax.random.permutation(rperm, B)
        mixed_z = project_z(self._apply("phi", self.params["phi"], goal)[perm], c["norm_z"])
        mix_mask = (jax.random.uniform(r5, (B,)) < c["mix_ratio"])[:, None]
        z_cont = jnp.where(mix_mask, mixed_z, gauss_z)
        zbin = (jax.random.randint(r2, (B,), 0, 2 ** c["max_log_seed"])[:, None]
                & (1 << jnp.arange(c["max_log_seed"]))) > 0
        z_psm = zbin.astype(jnp.float32)
        seed_to_action, powers, max_seed = self.proto
        obs_hash = jnp.arange(B)
        proto_na = proto_sample(seed_to_action, powers, obs_hash, z_psm, max_seed)
        mu_next = self._apply("actor", self.params["actor"], batch["next_observations"], z_cont)
        sf_na = truncated_sample(mu_next, c["actor_std"], jax.random.normal(r3, (B, adim)), clip=c["stddev_clip"])
        mu = self._apply("actor", self.params["actor"], batch["observations"], z_cont)
        actor_smp = truncated_sample(mu, c["actor_std"], jax.random.normal(r4, (B, adim)), clip=c["stddev_clip"])
        return dict(z_cont=z_cont, z_psm=z_psm, proto_next_action=proto_na,
                    actor_next_action=sf_na, actor_sample=actor_smp)

    def update(self, batch):
        # NOTE: not @jax.jit yet — nets/txs/proto are dict-valued static fields that
        # are not hashable for jit's static aux. Functionally correct; jitting (via
        # a ModuleDict/FrozenDict refactor) is a perf follow-up.
        new_rng, rng = jax.random.split(self.rng)
        inj = self._draw_injection(batch, rng)
        new_agent, info = self.apply_update(batch, inj)
        return new_agent.replace(rng=new_rng), info

    def sample_actions(self, observations, seed=None, temperature=1.0):
        # PSM acts with the TD3 actor mean; z must be supplied for goal-conditioned
        # acting. For OGBench single-task eval we infer z from rewards (see infer_z);
        # here we use a zero z as a placeholder when none is provided.
        z = jnp.zeros((*observations.shape[:-1], self.config["z_dim"]))
        return self._apply("actor", self.params["actor"], observations, z)

    def total_loss(self, batch, grad_params=None, rng=None):
        """Validation-logging loss: the three branch losses at current params (no step)."""
        rng = rng if rng is not None else self.rng
        inj = self._draw_injection(batch, rng)
        B = batch["observations"].shape[0]
        off, off_sum = self._off(B)
        psm_fn, sf_fn, actor_fn = self._stages(self.params, batch, inj, off, off_sum)
        psm_l, psm_i = psm_fn(self.params["phi"], self.params["psm_psi"])
        sf_l, sf_i = sf_fn(self.params["sf_psi"], self.params["phi"])
        a_l, a_i = actor_fn(self.params["actor"], self.params["sf_psi"])
        return psm_l + sf_l + a_l, {**psm_i, **sf_i, **a_i}

    def infer_z(self, next_observations, rewards):
        phi = self._apply("phi", self.params["phi"], next_observations)
        z = (rewards.reshape(1, -1) @ phi).reshape(-1) / phi.shape[0]
        return project_z(z, self.config["norm_z"])

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, rphi, rsf, rpsm, ract, rproto = jax.random.split(rng, 6)
        obs_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]
        z_dim = config["z_dim"]

        nets = dict(
            phi=PhiMap(z_dim=z_dim, hidden_dim=config["phi"]["hidden_dim"],
                       hidden_layers=config["phi"]["hidden_layers"], norm=True),
            sf_psi=PsiMap(output_dim=z_dim, hidden_dim=config["sf"]["hidden_dim"],
                          num_parallel=config["num_parallel"],
                          embedding_layers=config["sf"]["embedding_layers"],
                          hidden_layers=config["sf"]["hidden_layers"]),
            psm_psi=PsiMap(output_dim=z_dim, hidden_dim=config["sf"]["hidden_dim"],
                           num_parallel=config["num_parallel"],
                           embedding_layers=config["sf"]["embedding_layers"],
                           hidden_layers=config["sf"]["hidden_layers"]),
            actor=PSMActor(action_dim=action_dim, hidden_dim=config["actor"]["hidden_dim"],
                           embedding_layers=config["actor"]["embedding_layers"],
                           hidden_layers=config["actor"]["hidden_layers"]),
        )
        ex_obs = ex_observations
        ex_z = jnp.zeros((ex_obs.shape[0], z_dim))
        ex_zbin = jnp.zeros((ex_obs.shape[0], config["max_log_seed"]))
        params = {
            "phi": nets["phi"].init(rphi, ex_obs)["params"],
            "sf_psi": nets["sf_psi"].init(rsf, ex_obs, ex_z, ex_actions)["params"],
            "psm_psi": nets["psm_psi"].init(rpsm, ex_obs, ex_zbin, ex_actions)["params"],
            "actor": nets["actor"].init(ract, ex_obs, ex_z)["params"],
        }
        params["target_phi"] = copy.deepcopy(params["phi"])
        params["target_psm_psi"] = copy.deepcopy(params["psm_psi"])
        params["target_sf_psi"] = copy.deepcopy(params["sf_psi"])

        txs = dict(
            phi=optax.adam(config["lr_phi"]), psm_psi=optax.adam(config["lr_sf"]),
            sf_psi=optax.adam(config["lr_sf"]), actor=optax.adam(config["lr_actor"]),
        )
        opt_states = {k: txs[k].init(params[k]) for k in ["phi", "psm_psi", "sf_psi", "actor"]}

        # proto table (jax-generated; used only for training — tests inject it)
        max_seed = 2 ** config["max_log_seed"] + 20000
        table = (jax.random.uniform(rproto, (max_seed, action_dim)) - 1.0) * 2.0
        powers = (2 ** jnp.arange(config["max_log_seed"]))[::-1].astype(jnp.float32)
        proto = (table.astype(jnp.float32), powers, max_seed)

        config = dict(config)
        config["ob_dims"] = list(ex_observations.shape[1:])
        config["action_dim"] = action_dim
        return cls(rng=rng, params=params, opt_states=opt_states,
                   config=flax.core.FrozenDict(config), nets=nets, txs=txs, proto=proto)


def _step(tx, grad, params, opt_state):
    updates, new_opt = tx.update(grad, opt_state, params)
    return optax.apply_updates(params, updates), new_opt


def _soft(online, target, tau):
    return jax.tree_util.tree_map(lambda p, tp: p * tau + tp * (1 - tau), online, target)
