"""Affine (full) PSM agent — M(s,a,x) = Phi(s,a,x)·w + b(s,a,x).

Faithful JAX port of RLU controllable_agent/url_benchmark/agent/psm.py (continuous)
+ discrete_psm.py `_infer_step`. Distinct from the bilinear agents/psm.py: the task
coordinate w enters LINEARLY, which is what makes the constrained-LP `full` inference
well-defined. See docs/superpowers/specs/2026-07-22-affine-psm-design.md.
"""
import copy
from typing import Any

import flax
import flax.linen as fnn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from utils.flax_utils import TrainState, nonpytree_field
from utils.psm_networks import AffineMeasureNet, LagrangeNet, PSMActor
from agents.psm import proto_sample, project_z, off_diagonal_mask, polyak_update, _plain_config


class _WNet(fnn.Module):
    """z (codebook code) -> w (task coordinates), an MLP (RLU psm.py self.w)."""
    d_dim: int
    hidden_dim: int

    @fnn.compact
    def __call__(self, z):
        h = z
        for _ in range(3):
            h = fnn.relu(fnn.Dense(self.hidden_dim)(h))
        return fnn.Dense(self.d_dim)(h)


@flax.struct.dataclass
class StepInputs:
    proto_seed: Any
    proto_next_action: Any


class AffinePSMAgent(flax.struct.PyTreeNode):
    rng: Any
    measure: TrainState
    w: TrainState
    target_measure: Any
    target_w: Any
    w_inf: Any                 # (d_dim,) inference task coordinate, solved at eval
    actor: TrainState          # PSMActor distilled against Q = Phi·w_inf + b
    config: Any = nonpytree_field()
    proto: Any = None          # (seed_to_action, powers)

    def sample_step_inputs(self, batch, rng):
        c = self.config
        B = batch["observations"].shape[0]
        r_seed = jax.random.fold_in(rng, 0)
        # ONE binary codebook code per batch (RLU samples one z and repeats it).
        w = c["max_log_seed"]
        code = jax.random.randint(r_seed, (), 0, 2 ** w)
        bits = ((code >> jnp.arange(w)) & 1).astype(jnp.float32)
        proto_seed = jnp.broadcast_to(bits, (B, w))
        seed_to_action, powers = self.proto
        obs_hash = batch["index"] if "index" in batch else jnp.arange(B)
        proto_next_action = proto_sample(seed_to_action, powers, obs_hash, proto_seed, c["proto_max_seed"])
        return StepInputs(proto_seed=proto_seed, proto_next_action=proto_next_action)

    def measure_loss(self, batch, sampled, measure_params, w_params):
        c = self.config
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        x = next_obs                      # measure argument (future state)
        B = obs.shape[0]
        off, off_sum = off_diagonal_mask(B)
        z, proto_na = sampled.proto_seed, sampled.proto_next_action

        # B^2 mesh over (s_i, a_i, x_j): i indexes the source (s,a), j the measure arg x.
        i_idx = jnp.repeat(jnp.arange(B), B)     # (B^2,)
        j_idx = jnp.tile(jnp.arange(B), B)       # (B^2,)
        m_obs, m_act = obs[i_idx], action[i_idx]
        m_next_obs, m_next_act = next_obs[i_idx], proto_na[i_idx]
        m_x = x[j_idx]

        wz = self.w(z, params=w_params)          # (B, d) — identical rows
        wz_i = wz[i_idx]
        phi, b = self.measure(m_obs, m_act, m_x, params=measure_params)
        M = ((phi * wz_i).sum(-1, keepdims=True) + b).reshape(B, B)

        tphi, tb = self.measure(m_next_obs, m_next_act, m_x, params=self.target_measure)
        twz = self.w(z, params=self.target_w)[i_idx]
        target_M = ((tphi * twz).sum(-1, keepdims=True) + tb).reshape(B, B)
        target_M = jax.lax.stop_gradient(target_M)

        diff = M - c["discount"] * target_M
        offdiag = 0.5 * jnp.sum((diff * off) ** 2) / off_sum
        # RLU psm.py:352 diagonal source term: -(1-gamma)*mean(diag(M)).
        diag = -((1 - c["discount"]) * jnp.diagonal(M)).mean()
        loss = offdiag + diag
        return loss, {"psm_loss": loss, "psm_diag": diag, "psm_offdiag": offdiag}

    def apply_update(self, batch, sampled):
        tau = self.config["tau"]
        (_, info), (g_m, g_w) = jax.value_and_grad(self.measure_loss, argnums=(2, 3), has_aux=True)(
            batch, sampled, self.measure.params, self.w.params)
        measure = self.measure.apply_gradients(grads=g_m)
        w = self.w.apply_gradients(grads=g_w)
        target_measure = polyak_update(measure.params, self.target_measure, tau)
        target_w = polyak_update(w.params, self.target_w, tau)
        return self.replace(measure=measure, w=w, target_measure=target_measure, target_w=target_w), info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        sampled = self.sample_step_inputs(batch, rng)
        new_agent, info = self.apply_update(batch, sampled)
        return new_agent.replace(rng=new_rng), info

    def total_loss(self, batch, grad_params=None, rng=None):
        rng = rng if rng is not None else self.rng
        sampled = self.sample_step_inputs(batch, rng)
        loss, info = self.measure_loss(batch, sampled, self.measure.params, self.w.params)
        return loss, info

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, rm, rw, ract, rproto = jax.random.split(rng, 5)
        assert config.get("encoder", None) is None, "affine_psm does not support visual encoders."
        action_dim = ex_actions.shape[-1]
        d_dim, z_dim = config["d_dim"], config["z_dim"]
        ex_obs, ex_act = ex_observations, ex_actions
        ex_x = ex_obs
        ex_z = jnp.zeros((ex_obs.shape[0], z_dim))          # actor task-conditioning (zeroed placeholder)
        ex_zbin = jnp.zeros((ex_obs.shape[0], config["max_log_seed"]))  # binary codebook code -> w

        measure_def = AffineMeasureNet(d_dim=d_dim, hidden_dim=config["measure"]["hidden_dim"],
                                       hidden_layers=config["measure"]["hidden_layers"])
        measure = TrainState.create(measure_def, measure_def.init(rm, ex_obs, ex_act, ex_x)["params"],
                                    tx=optax.adam(config["lr"]))

        # w: binary codebook code (max_log_seed bits) -> d task coords (RLU psm.py self.w).
        w_def = _WNet(d_dim=d_dim, hidden_dim=config["measure"]["hidden_dim"])
        w = TrainState.create(w_def, w_def.init(rw, ex_zbin)["params"], tx=optax.adam(config["lr_w"]))

        actor_cfg = config["actor"]
        actor_def = PSMActor(action_dim=action_dim, hidden_dim=actor_cfg["hidden_dim"],
                             embedding_layers=actor_cfg["embedding_layers"],
                             hidden_layers=actor_cfg["hidden_layers"])
        actor = TrainState.create(actor_def, actor_def.init(ract, ex_obs, ex_z)["params"],
                                  tx=optax.adam(config["lr"]))

        # proto codebook table (reuse the psm.py scheme).
        max_seed = 2 ** config["max_log_seed"] + 20000
        table = (jax.random.uniform(rproto, (max_seed, action_dim)) - 1.0) * 2.0
        powers = (2 ** jnp.arange(config["max_log_seed"]))[::-1].astype(jnp.float32)
        proto = (table.astype(jnp.float32), powers)

        config = _plain_config(config)
        config["ob_dims"] = tuple(ex_observations.shape[1:])
        config["action_dim"] = action_dim
        config["proto_max_seed"] = max_seed
        return cls(rng=rng, measure=measure, w=w,
                   target_measure=copy.deepcopy(measure.params),
                   target_w=copy.deepcopy(w.params),
                   w_inf=jnp.ones((d_dim,), jnp.float32), actor=actor,
                   config=flax.core.FrozenDict(config), proto=proto)
