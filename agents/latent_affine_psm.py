"""Affine PSM over the behavior flow's LATENT action space.

`agents/affine_psm.py` is the full PSM on raw actions. This agent substitutes the latent u
for the action in the measure's action slot; an action appears only at act time, by
decoding u through the FROZEN behavior flow, so every expressible policy is a flow decode.
Single latent per transition by construction.

Requires a preimage-augmented dataset (tools/precompute_preimages.py) and the Stage-A
checkpoint it was inverted from; main.py loads the npz and asserts the pairing.
"""
from typing import Any

import jax
import jax.numpy as jnp

from agents.affine_psm import AffinePSMAgent
from utils.flax_utils import nonpytree_field
from utils.networks import ActorVectorField


class LatentAffinePSMAgent(AffinePSMAgent):
    """Affine PSM whose action slot is the frozen flow's latent space."""

    # Frozen behaviour flow. Params are pytree leaves (they never receive gradients, no
    # TrainState, no optimizer); the module defs are static aux like psmflow's.
    flow_vf: Any = None
    flow_onestep: Any = None
    flow_vf_def: Any = nonpytree_field(default=None)
    flow_onestep_def: Any = nonpytree_field(default=None)

    # ---- action-slot seams (see agents/affine_psm.py) ----
    def measure_action(self, batch):
        """The dataset latent u = E(s, a), clipped to the typical-set box.

        The clip is the same one psmflow applies: the actor side is tanh-bounded to
        +-u_clip, so an unclipped u_data would hand the online branch inputs the target
        branch can never produce.
        """
        return jnp.clip(jnp.asarray(batch["noise_preimage"]),
                        -self.config["u_clip"], self.config["u_clip"])

    def action_scale(self):
        return self.config["u_clip"]

    def to_env_action(self, observations, u):
        """Decode a latent through the FROZEN flow. This is the only place an action
        appears; everything upstream is latent-space."""
        if self.config["gpi_decode"] == "onestep":
            a = self.flow_onestep_def.apply({"params": self.flow_onestep}, observations, u)
        else:
            a = u
            steps = self.config["flow_decode_steps"]
            for i in range(steps):
                t = jnp.full((*observations.shape[:-1], 1), i / steps)
                a = a + self.flow_vf_def.apply({"params": self.flow_vf}, observations, a, t) / steps
        return jnp.clip(a, -1.0, 1.0)

    @staticmethod
    def codebook_table(rng, max_seed, dim, config):
        """Codebook of LATENT policies, drawn from the flow's own N(0, I) prior.

        The reference table is uniform in [-2, 0) because its entries are actions; here an
        entry is a latent the flow will decode, so the prior it was trained under is the
        right draw. pi_z(s) = table[(z·powers + hash(s)) mod max_seed] varies with the
        state, so this is NOT the fixed-u family (HANDOFF 2026-08-05).
        """
        u_clip = float(config["u_clip"])
        return jnp.clip(jax.random.normal(rng, (max_seed, dim)), -u_clip, u_clip)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        from agents.psmflow import _load_flow_params

        action_dim = ex_actions.shape[-1]
        flow_hidden = tuple(config["flow"]["hidden_dims"])
        vf_def = ActorVectorField(hidden_dims=flow_hidden, action_dim=action_dim,
                                  layer_norm=config["flow"]["layer_norm"])
        onestep_def = ActorVectorField(hidden_dims=flow_hidden, action_dim=action_dim,
                                       layer_norm=config["flow"]["layer_norm"])
        ckpt = config.get("flow_ckpt_path", None)
        if ckpt:
            flow_vf, flow_onestep = _load_flow_params(
                ckpt, config.get("flow_ckpt_epoch", None), config, ex_observations, ex_actions)
        else:
            assert config.get("allow_untrained_flow", False), (
                "latent_affine_psm requires agent.flow_ckpt_path (a Stage-A fql bc_only "
                "ckpt dir); set agent.allow_untrained_flow=true only for tests/smokes.")
            rvf, ronestep = jax.random.split(jax.random.PRNGKey(seed + 7919))
            ex_times = ex_actions[..., :1]
            flow_vf = vf_def.init(rvf, ex_observations, ex_actions, ex_times)["params"]
            flow_onestep = onestep_def.init(ronestep, ex_observations, ex_actions)["params"]

        # The base builder returns an instance of THIS class (it calls `cls(...)`), so the
        # flow fields take their declared defaults and are filled in here.
        agent = super().create(seed, ex_observations, ex_actions, config)
        return agent.replace(flow_vf=flow_vf, flow_onestep=flow_onestep,
                             flow_vf_def=vf_def, flow_onestep_def=onestep_def)
