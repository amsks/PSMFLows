"""FB (Forward-Backward) agent — JAX/Flax port of the PyTorch factored-fb reference.

FB is PSM with the proto branch removed: one Forward map F(left_enc(obs), z, action), one
Backward map B(next_obs) (the measure basis AND the z-source), a left_encoder trunk, and a
td3/flow actor. M = F.B^T, trained with the same off-diag/diag + ortho loss as PSM.

Cube-default path only: measure critic, perm goal-mode, onestep off, penalties 0. Reuses
PSM's scaffolding so per-step numerics stay testable.
"""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from utils.flax_utils import nonpytree_field
from utils.fb_networks import BackwardMap, ForwardMap, FBTd3Actor
from utils.psm_networks import NoiseConditionedActor, FlowVectorField, truncated_sample
# Reuse PSM's shared helpers verbatim (identical math) to avoid duplication.
from agents.psm import (
    _HashableDict, _plain_config, _step, _soft, project_z,
    contrastive_loss, ortho_loss, targets_uncertainty,
)


class FBAgent(flax.struct.PyTreeNode):
    rng: Any
    params: Any        # forward/backward/left_encoder/actor(+actor_vf) + target_{forward,backward,left_encoder}
    opt_states: Any    # forward/left_encoder/backward/actor(+actor_vf)
    z_eval: Any        # (z_dim,) task latent for sample_actions; set via infer_eval_z
    config: Any = nonpytree_field()
    nets: Any = nonpytree_field()
    txs: Any = nonpytree_field()

    def _apply(self, name, params, *args):
        return self.nets[name].apply({"params": params}, *args)

    def _off(self, B):
        off = 1.0 - jnp.eye(B)
        return off, off.sum()

    # -------------------------- loss stages --------------------------
    def _fb_loss_fn(self, params, batch, inj, off, off_sum):
        """FB measure loss over (forward, left_encoder, backward). z and next_action are
        injected constants (stop-grad). Targets read from params['target_*']."""
        c = self.config
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        goal = next_obs  # bw_encoder = Identity for state
        # Reference forces terminated=False for every cube transition (data/ogbench.py)
        # => always-gamma bootstrap. OGBench's packaged `terminals`=1 at each episode's
        # final step would otherwise cut the bootstrap there; force always-gamma to match
        # the reference (same fix as PSM's masks=always-gamma). Scalar broadcasts over M.
        disc = c["discount"]
        P = c["num_parallel"]
        z = inj["z"]
        next_action = inj["next_action"]

        def loss_fn(fwd_p, le_p, bwd_p):
            # target measure (stop-grad): target nets on next_obs.
            nle = self._apply("left_encoder", params["target_left_encoder"], next_obs)
            tFs = self._apply("forward", params["target_forward"], nle, z, next_action)
            tB = self._apply("backward", params["target_backward"], goal)
            tM = jnp.einsum("pbz,cz->pbc", tFs, tB)
            tmean, tunc = targets_uncertainty(tM, P)
            target_M = jax.lax.stop_gradient(tmean - c["fb_pessimism_penalty"] * tunc)
            # online measure.
            le = self._apply("left_encoder", le_p, obs)
            Fs = self._apply("forward", fwd_p, le, z, action)
            B = self._apply("backward", bwd_p, goal)
            M = jnp.einsum("pbz,cz->pbc", Fs, B)
            diff = M - disc * target_M
            fb_off = 0.5 * jnp.sum((diff * off) ** 2) / off_sum
            fb_diag = -jnp.mean(jnp.diagonal(diff, axis1=1, axis2=2)) * P
            cov = B @ B.T
            orth_off = 0.5 * jnp.sum((cov * off) ** 2) / off_sum
            orth_diag = -jnp.mean(jnp.diagonal(cov))
            orth = orth_off + orth_diag
            loss = fb_off + fb_diag + c["ortho_coef"] * orth
            return loss, {"fb_loss": loss, "fb_offdiag": fb_off, "fb_diag": fb_diag,
                          "ortho_loss": orth}
        return loss_fn

    def _actor_loss_fn(self, params, batch, inj, off, off_sum):
        """Actor loss over (actor[, actor_vf]). Q = <F(left_enc(obs), z, a), z>; F/left_enc
        read as stop-grad constants. Flow: + flow-matching BC + one-step distillation."""
        c = self.config
        obs, action = batch["observations"], batch["actions"]
        z = inj["z"]
        P = c["num_parallel"]
        # left_enc is detached in the reference actor update.
        left_enc = jax.lax.stop_gradient(self._apply("left_encoder", params["left_encoder"], obs))

        def q_of(a):
            Fs = self._apply("forward", params["forward"], left_enc, z, a)
            Qs = (Fs * z).sum(-1)
            qmean, qunc = targets_uncertainty(Qs, P)
            return qmean - c["actor_pessimism_penalty"] * qunc, Qs

        if c["actor_type"] == "flow":
            x0, t, noise = inj["flow_x0"], inj["flow_t"], inj["actor_noise"]
            steps, bc_coeff = c["flow_steps"], c["bc_coeff"]

            def rollout(vf_p, o, n):
                a = n
                for i in range(steps):
                    ti = jnp.full((o.shape[0], 1), i / steps)
                    a = a + self._apply("actor_vf", vf_p, o, a, ti) / steps
                return jnp.clip(a, -1.0, 1.0)

            def loss_fn(actor_p, vf_p):
                x1 = action
                xt = (1 - t) * x0 + t * x1
                vel = x1 - x0
                pred = self._apply("actor_vf", vf_p, obs, xt, t)
                bc_flow = jnp.mean((pred - vel) ** 2)
                a = self._apply("actor", actor_p, obs, z, noise)
                Q, Qs = q_of(a)
                loss = -Q.mean()
                info = {"q": Q.mean(), "bc_flow_loss": bc_flow}
                if bc_coeff > 0:
                    target = jax.lax.stop_gradient(rollout(vf_p, obs, noise))
                    distill = jnp.mean((a - target) ** 2)
                    loss = loss / jax.lax.stop_gradient(jnp.abs(Qs).mean()) + bc_coeff * distill
                    info["bc_error"] = distill
                loss = loss + bc_flow
                info["actor_loss"] = loss
                return loss, info
            return loss_fn, ("actor", "actor_vf")

        # td3 actor
        def loss_fn_td3(actor_p):
            mu = self._apply("actor", actor_p, obs, z)
            a = truncated_sample(mu, c["actor_std"], inj["actor_noise"], clip=c["stddev_clip"])
            Q, Qs = q_of(a)
            loss = -Q.mean()
            info = {"actor_loss": loss, "q": Q.mean()}
            if c["bc_coeff"] > 0:
                bc_error = jnp.mean((a - action) ** 2)
                loss = loss / jax.lax.stop_gradient(jnp.abs(Qs).mean()) + c["bc_coeff"] * bc_error
                info = {"actor_loss": loss, "q": Q.mean(), "bc_error": bc_error}
            return loss, info
        return loss_fn_td3, ("actor",)

    # -------------------------- test hooks --------------------------
    def compute_static(self, batch, inj):
        """FB + actor losses/grads at CURRENT params, no interleaving (matches the
        fixture's no-step static export)."""
        B = batch["observations"].shape[0]
        off, off_sum = self._off(B)
        fb_fn = self._fb_loss_fn(self.params, batch, inj, off, off_sum)
        (fb_l, fb_i), (g_fwd, g_le, g_bwd) = jax.value_and_grad(fb_fn, argnums=(0, 1, 2), has_aux=True)(
            self.params["forward"], self.params["left_encoder"], self.params["backward"])
        actor_fn, akeys = self._actor_loss_fn(self.params, batch, inj, off, off_sum)
        if akeys == ("actor", "actor_vf"):
            (a_l, a_i), (g_a, g_vf) = jax.value_and_grad(actor_fn, argnums=(0, 1), has_aux=True)(
                self.params["actor"], self.params["actor_vf"])
        else:
            (a_l, a_i), g_a = jax.value_and_grad(actor_fn, has_aux=True)(self.params["actor"])
        return {**fb_i, **a_i}

    def apply_update(self, batch, inj):
        """2-stage update: FB (forward/left_encoder/backward) then actor; soft-update targets."""
        B = batch["observations"].shape[0]
        off, off_sum = self._off(B)
        params = dict(self.params)
        opt = dict(self.opt_states)
        f_tau, b_tau = self.config["f_target_tau"], self.config["b_target_tau"]

        # stage 1: FB loss -> step forward, left_encoder, backward
        fb_fn = self._fb_loss_fn(params, batch, inj, off, off_sum)
        (fb_l, fb_i), (g_fwd, g_le, g_bwd) = jax.value_and_grad(fb_fn, argnums=(0, 1, 2), has_aux=True)(
            params["forward"], params["left_encoder"], params["backward"])
        params["forward"], opt["forward"] = _step(self.txs["forward"], g_fwd, params["forward"], opt["forward"])
        params["left_encoder"], opt["left_encoder"] = _step(self.txs["left_encoder"], g_le, params["left_encoder"], opt["left_encoder"])
        params["backward"], opt["backward"] = _step(self.txs["backward"], g_bwd, params["backward"], opt["backward"])

        # stage 2: actor loss -> step actor (+ vf)
        actor_fn, akeys = self._actor_loss_fn(params, batch, inj, off, off_sum)
        if akeys == ("actor", "actor_vf"):
            (a_l, a_i), (g_a, g_vf) = jax.value_and_grad(actor_fn, argnums=(0, 1), has_aux=True)(
                params["actor"], params["actor_vf"])
            params["actor"], opt["actor"] = _step(self.txs["actor"], g_a, params["actor"], opt["actor"])
            params["actor_vf"], opt["actor_vf"] = _step(self.txs["actor_vf"], g_vf, params["actor_vf"], opt["actor_vf"])
        else:
            (a_l, a_i), g_a = jax.value_and_grad(actor_fn, has_aux=True)(params["actor"])
            params["actor"], opt["actor"] = _step(self.txs["actor"], g_a, params["actor"], opt["actor"])

        # stage 3: soft-update targets (forward/left_encoder -> f_tau, backward -> b_tau).
        params["target_forward"] = _soft(params["forward"], params["target_forward"], f_tau)
        params["target_left_encoder"] = _soft(params["left_encoder"], params["target_left_encoder"], f_tau)
        params["target_backward"] = _soft(params["backward"], params["target_backward"], b_tau)

        return self.replace(params=params, opt_states=opt), {**fb_i, **a_i}

    # -------------------------- live update --------------------------
    def _draw_injection(self, batch, rng):
        c = self.config
        B = batch["observations"].shape[0]
        adim = c["action_dim"]
        next_obs = batch["next_observations"]
        r_z, r_mask, r_perm, rtail = jax.random.split(rng, 4)
        z_gauss = project_z(jax.random.normal(r_z, (B, c["z_dim"])), c["norm_z"])
        # perm goal-mode: mix B(next_obs[perm]) into z at train_goal_ratio.
        goals = project_z(self._apply("backward", self.params["backward"], next_obs)[
            jax.random.permutation(r_perm, B)], c["norm_z"])
        mix_mask = (jax.random.uniform(r_mask, (B,)) < c["train_goal_ratio"])[:, None]
        z = jnp.where(mix_mask, goals, z_gauss)
        inj = dict(z=z)
        if c["actor_type"] == "flow":
            r_na, r_x0, r_t, r_noise = jax.random.split(rtail, 4)
            na = self._apply("actor", self.params["actor"], next_obs, z,
                             jax.random.normal(r_na, (B, adim)))
            inj.update(next_action=na,
                       flow_x0=jax.random.normal(r_x0, (B, adim)),
                       flow_t=jax.random.uniform(r_t, (B, 1)),
                       actor_noise=jax.random.normal(r_noise, (B, adim)))
        else:
            r_na, r_as = jax.random.split(rtail)
            mu_next = self._apply("actor", self.params["actor"], next_obs, z)
            na = truncated_sample(mu_next, c["actor_std"], jax.random.normal(r_na, (B, adim)),
                                  clip=c["stddev_clip"])
            inj.update(next_action=na, actor_noise=jax.random.normal(r_as, (B, adim)))
        return inj

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        inj = self._draw_injection(batch, rng)
        new_agent, info = self.apply_update(batch, inj)
        return new_agent.replace(rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        z = jnp.broadcast_to(self.z_eval, (*observations.shape[:-1], self.config["z_dim"]))
        if self.config["actor_type"] == "flow":
            seed = self.rng if seed is None else seed
            noise = jax.random.normal(seed, (*observations.shape[:-1], self.config["action_dim"]))
            a = self._apply("actor", self.params["actor"], observations, z, noise)
            return jnp.clip(a, -1.0, 1.0)
        return self._apply("actor", self.params["actor"], observations, z)

    def total_loss(self, batch, grad_params=None, rng=None):
        rng = rng if rng is not None else self.rng
        inj = self._draw_injection(batch, rng)
        B = batch["observations"].shape[0]
        off, off_sum = self._off(B)
        fb_l, fb_i = self._fb_loss_fn(self.params, batch, inj, off, off_sum)(
            self.params["forward"], self.params["left_encoder"], self.params["backward"])
        actor_fn, akeys = self._actor_loss_fn(self.params, batch, inj, off, off_sum)
        if akeys == ("actor", "actor_vf"):
            a_l, a_i = actor_fn(self.params["actor"], self.params["actor_vf"])
        else:
            a_l, a_i = actor_fn(self.params["actor"])
        return fb_l + a_l, {**fb_i, **a_i}

    def infer_z(self, next_observations, rewards):
        B = self._apply("backward", self.params["backward"], next_observations)
        z = (rewards.reshape(1, -1) @ B).reshape(-1) / B.shape[0]
        return project_z(z, self.config["norm_z"])

    def infer_eval_z(self, next_observations, rewards):
        return self.replace(z_eval=self.infer_z(next_observations, rewards))

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, rf, rb, rle, ract = jax.random.split(rng, 5)
        assert config.get("encoder", None) is None, "FB does not support visual encoders yet."
        action_dim = ex_actions.shape[-1]
        z_dim = config["z_dim"]
        L_dim = config["L_dim"]
        actor_cfg = config["actor"]
        actor_type = actor_cfg["type"] if "type" in actor_cfg else "flow"
        assert actor_type in ("td3", "flow"), f"unknown actor.type {actor_type!r}"
        P = config["num_parallel"]

        nets = dict(
            forward=ForwardMap(z_dim=z_dim, hidden_dim=config["forward"]["hidden_dim"],
                               num_parallel=P, hidden_layers=config["forward"]["hidden_layers"],
                               embedding_layers=config["forward"]["embedding_layers"]),
            backward=BackwardMap(z_dim=z_dim, hidden_dim=config["backward"]["hidden_dim"],
                                 hidden_layers=config["backward"]["hidden_layers"],
                                 norm=config["backward"]["norm"]),
            left_encoder=BackwardMap(z_dim=L_dim, hidden_dim=config["left_encoder"]["hidden_dim"],
                                     hidden_layers=config["left_encoder"]["hidden_layers"],
                                     norm=config["left_encoder"]["norm"]),
        )
        ex_obs = ex_observations
        ex_z = jnp.zeros((ex_obs.shape[0], z_dim))
        ex_feat = jnp.zeros((ex_obs.shape[0], L_dim))
        params = {
            "left_encoder": nets["left_encoder"].init(rle, ex_obs)["params"],
            "backward": nets["backward"].init(rb, ex_obs)["params"],
            "forward": nets["forward"].init(rf, ex_feat, ex_z, ex_actions)["params"],
        }

        actor_keys = ["actor"]
        if actor_type == "flow":
            def _acfg(key, default):
                return actor_cfg[key] if key in actor_cfg else default
            fa_hidden = int(_acfg("flow_actor_hidden_dim", 512))
            fa_layers = int(_acfg("flow_actor_hidden_layers", 2))
            fa_emb = int(_acfg("flow_actor_embedding_layers", 2))
            vf_hidden = int(_acfg("flow_vf_hidden_dim", 512))
            vf_layers = int(_acfg("flow_vf_hidden_layers", 4))
            ract, rvf = jax.random.split(ract)
            nets["actor"] = NoiseConditionedActor(action_dim=action_dim, hidden_dim=fa_hidden,
                                                  hidden_layers=fa_layers, embedding_layers=fa_emb)
            nets["actor_vf"] = FlowVectorField(action_dim=action_dim, hidden_dim=vf_hidden,
                                               hidden_layers=vf_layers)
            params["actor"] = nets["actor"].init(ract, ex_obs, ex_z, ex_actions)["params"]
            params["actor_vf"] = nets["actor_vf"].init(rvf, ex_obs, ex_actions, ex_actions[..., :1])["params"]
            actor_keys = ["actor", "actor_vf"]
        else:
            nets["actor"] = FBTd3Actor(action_dim=action_dim, hidden_dim=actor_cfg["hidden_dim"],
                                       embedding_layers=actor_cfg["embedding_layers"],
                                       hidden_layers=actor_cfg["hidden_layers"])
            params["actor"] = nets["actor"].init(ract, ex_obs, ex_z)["params"]

        params["target_forward"] = copy.deepcopy(params["forward"])
        params["target_backward"] = copy.deepcopy(params["backward"])
        params["target_left_encoder"] = copy.deepcopy(params["left_encoder"])

        lr_actor_vf = float(actor_cfg["lr_actor_vf"]) if "lr_actor_vf" in actor_cfg else 3e-4
        txs = dict(
            forward=optax.adam(config["lr_f"]), left_encoder=optax.adam(config["lr_f"]),
            backward=optax.adam(config["lr_b"]), actor=optax.adam(config["lr_actor"]),
        )
        if actor_type == "flow":
            txs["actor_vf"] = optax.adam(lr_actor_vf)
        opt_states = {k: txs[k].init(params[k])
                      for k in ["forward", "left_encoder", "backward", *actor_keys]}

        config = _plain_config(config)
        config["ob_dims"] = tuple(ex_observations.shape[1:])
        config["action_dim"] = action_dim
        config["actor_type"] = actor_type
        config["bc_coeff"] = float(actor_cfg["bc_coeff"]) if "bc_coeff" in actor_cfg else 0.0
        config["flow_steps"] = int(actor_cfg["flow_steps"]) if "flow_steps" in actor_cfg else 10
        config["lr_actor_vf"] = lr_actor_vf
        return cls(rng=rng, params=params, opt_states=opt_states,
                   z_eval=jnp.zeros((z_dim,), jnp.float32),
                   config=flax.core.FrozenDict(config),
                   nets=_HashableDict(nets), txs=_HashableDict(txs))


def get_config():
    import ml_collections
    return ml_collections.ConfigDict(dict(
        agent_name="fb", batch_size=256, z_dim=50, L_dim=50, num_parallel=2,
        discount=0.99, f_target_tau=0.005, b_target_tau=0.005, ortho_coef=1.0,
        train_goal_ratio=0.5, fb_pessimism_penalty=0.0, actor_pessimism_penalty=0.0,
        actor_std=0.2, stddev_clip=0.3, norm_z=True, actor_encode_obs=False,
        weight_decay=0.0, lr_f=1.0e-4, lr_b=1.0e-4, lr_actor=1.0e-4,
        forward=dict(hidden_dim=512, hidden_layers=2, embedding_layers=2),
        backward=dict(hidden_dim=512, hidden_layers=4, norm=True),
        left_encoder=dict(hidden_dim=512, hidden_layers=4, norm=True),
        actor=dict(type="flow", hidden_dim=512, hidden_layers=2, embedding_layers=2,
                   bc_coeff=3.0, flow_steps=10, lr_actor_vf=3.0e-4,
                   flow_actor_hidden_dim=512, flow_actor_hidden_layers=2,
                   flow_actor_embedding_layers=2, flow_vf_hidden_dim=512,
                   flow_vf_hidden_layers=4),
        ob_dims=ml_collections.config_dict.placeholder(list),
        action_dim=ml_collections.config_dict.placeholder(int),
        encoder=ml_collections.config_dict.placeholder(str),
    ))
