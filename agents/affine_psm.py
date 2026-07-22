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

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, rm, rw, ract, rproto = jax.random.split(rng, 5)
        assert config.get("encoder", None) is None, "affine_psm does not support visual encoders."
        action_dim = ex_actions.shape[-1]
        d_dim, z_dim = config["d_dim"], config["z_dim"]
        ex_obs, ex_act = ex_observations, ex_actions
        ex_x = ex_obs
        ex_z = jnp.zeros((ex_obs.shape[0], z_dim))

        measure_def = AffineMeasureNet(d_dim=d_dim, hidden_dim=config["measure"]["hidden_dim"],
                                       hidden_layers=config["measure"]["hidden_layers"])
        measure = TrainState.create(measure_def, measure_def.init(rm, ex_obs, ex_act, ex_x)["params"],
                                    tx=optax.adam(config["lr"]))

        # w: z (codebook code) -> d task coords.
        w_def = _WNet(d_dim=d_dim, hidden_dim=config["measure"]["hidden_dim"])
        w = TrainState.create(w_def, w_def.init(rw, ex_z)["params"], tx=optax.adam(config["lr_w"]))

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
