"""Affine PSM over the behavior flow's LATENT action space.

`agents/affine_psm.py` is the full PSM: the affine measure M(s, a, x) = Phi(s,a,x)·w +
b(s,a,x), a basis learned against a hash codebook of policies, and the constrained-LP task
inference the affine form makes well-defined. It operates on raw environment actions, so
its measure is defined over the whole action space — including the actions the dataset
never contains.

This agent is the same PSM with ONE substitution: the measure's action slot carries the
flow's latent u instead of the action a, and an action is produced only at act time, by
decoding u through the FROZEN behavior flow. Every policy the measure can express is
therefore a flow decode and stays inside the behaviour the flow was cloned from, which is
the constraint the extra regularizers in offline PSM/FB exist to impose.

Single latent by construction: one u per transition (the Stage-B preimage), filling the
same slot the action filled. There is no second latent — the earlier `phi(s, u_0, u_0')`
form, which indexed the future rollout with its own latent, is not part of this agent.

Code <-> PSM:
  measure (FactoredAffineMeasureNet)  ->  M(s, u, x) = Phi(s,u,x)·w + b(s,u,x)
  w (WNet)                            ->  task coordinate for a codebook code z
  proto codebook                      ->  latent policies pi_z(s) = table[hash(z, s)],
                                          drawn from the flow's own N(0, I) prior
  actor                               ->  u(s, w), amortized, tanh * u_clip
  flow_vf / flow_onestep              ->  G_theta, FROZEN (FQL Stage-A checkpoint)
  batch['noise_preimage']             ->  u = E_theta(s, a), the dataset latent
  infer_eval                          ->  w_inf by LP (`full`) or closed form (`zero_shot`)

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

    # Frozen behaviour flow. Params are pytree leaves (they never receive gradients — no
    # TrainState, no optimizer); the module defs are static aux like psmflow's.
    flow_vf: Any = None
    flow_onestep: Any = None
    flow_vf_def: Any = nonpytree_field(default=None)
    flow_onestep_def: Any = nonpytree_field(default=None)

    # ---- action-slot seams (see agents/affine_psm.py) ----
    def _slot(self, batch):
        """The dataset latent u = E(s, a), clipped to the typical-set box.

        The clip is the same one psmflow applies: the actor side is tanh-bounded to
        +-u_clip, so an unclipped u_data would hand the online branch inputs the target
        branch can never produce.
        """
        return jnp.clip(jnp.asarray(batch["noise_preimage"]),
                        -self.config["u_clip"], self.config["u_clip"])

    def _slot_scale(self):
        return self.config["u_clip"]

    def _emit(self, observations, u):
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
    def _codebook_table(rng, max_seed, dim, config):
        """Codebook of LATENT policies: rows drawn from the flow's own N(0, I) prior,
        clipped to the same box as every other latent in the agent.

        RLU's table is uniform in [-2, 0) because its entries are actions. Here an entry
        is a latent the frozen flow will decode, so the prior it was trained under is the
        right draw — a uniform box would index policies the flow never saw.

        The codebook policy is pi_z(s) = table[(z·powers + hash(s)) mod max_seed], so the
        latent varies with the state. It is NOT a fixed-u policy: the fixed-u family was
        measured non-goal-covering (HANDOFF 2026-08-05), and this construction avoids it
        for the same reason the reference's does — the index selects a mapping, not a
        constant.
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
