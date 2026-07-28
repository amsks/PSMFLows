"""PSMFlow agent — successor measures over a flow-indexed policy family.

Code <-> research note (PAPER/RESEARCH_NOTE.md):
  phi (PhiMap)            -> varphi(x)        shared basis over future states
  psi (PsiMap)            -> psi(s, u', u)    measure head: z-slot = policy index u',
                                              action-slot = current latent u
  flow_vf / flow_onestep  -> G_theta          FROZEN behavior flow (FQL Stage-A ckpt)
  batch['noise_preimage'] -> u = E_theta(s,a) dataset latent (preimage pipeline)
  infer_z                 -> w = E[r varphi]  closed-form reward inference
  sample_actions          -> flow-GPI         argmax_u max_u' psi(s,u',u)^T w, decode

TD target: M^{u->u'}(s,.) backs up onto M^{u'->u'}(s',.) — the continuation latent IS
the index — so every action the backup implies is a flow decode (in-support, Prop. 3).
No decoded action appears anywhere in training; the flow is used only at act time.
"""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from agents.psm import (
    _plain_config, contrastive_loss, off_diagonal_mask, ortho_loss, polyak_update,
    targets_uncertainty,
)
from utils.flax_utils import TrainState, nonpytree_field
from utils.networks import ActorVectorField
from utils.psm_networks import PhiMap, PsiMap


@flax.struct.dataclass
class StepInputs:
    """Per-step sampled quantities.

    u_data:  (B, d_a) dataset latent (preimage of the batch action)
    u_index: (B, d_a) policy-index latent u' (mixed Gaussian/behavior, clipped)
    """
    u_data: Any
    u_index: Any


class PSMFlowAgent(flax.struct.PyTreeNode):
    rng: Any
    phi: TrainState
    psi: TrainState
    target_phi: Any
    target_psi: Any
    flow_vf: Any        # FROZEN params: multi-step BC velocity field (ODE decode)
    flow_onestep: Any   # FROZEN params: one-step distilled decoder (fast decode)
    task_z: Any         # (z_dim,) eval task vector w, set via infer_eval_z
    config: Any = nonpytree_field()
    flow_vf_def: Any = nonpytree_field(default=None)
    flow_onestep_def: Any = nonpytree_field(default=None)

    # ---- training ----
    def measure_loss(self, batch, sampled, phi_params, psi_params):
        c = self.config
        obs, next_obs = batch["observations"], batch["next_observations"]
        goal = next_obs  # phi_input='s' convention (as PSM)
        off, off_sum = off_diagonal_mask(obs.shape[0])
        P = c["num_parallel"]
        u, u_idx = sampled.u_data, sampled.u_index

        phi_g = self.phi(goal, params=phi_params)
        M = self.psi(obs, u_idx, u, params=psi_params) @ phi_g.T
        tphi = self.phi(goal, params=self.target_phi)
        # Continuation latent IS the index: the bootstrap policy at s' is pi_{u'}.
        tM = self.psi(next_obs, u_idx, u_idx, params=self.target_psi) @ tphi.T
        tmean, tunc = targets_uncertainty(tM, P)
        target_M = tmean - c["pessimism_penalty"] * tunc
        cl, cdiag, coff = contrastive_loss(M, jax.lax.stop_gradient(target_M), c["discount"], off, off_sum)
        ol, odiag, ooff = ortho_loss(phi_g, off, off_sum)
        loss = cl + c["ortho_coef"] * ol
        return loss, {"psm_loss": cl, "psm_diag": cdiag, "psm_offdiag": coff,
                      "orth_loss": ol, "orth_diag": odiag, "orth_offdiag": ooff}

    def sample_step_inputs(self, batch, rng):
        c = self.config
        B = batch["observations"].shape[0]
        u_data = jnp.asarray(batch["noise_preimage"])
        r_mix, r_gauss, r_perm = jax.random.split(rng, 3)
        gauss = jax.random.normal(r_gauss, (B, c["action_dim"]))
        perm = jax.random.permutation(r_perm, B)
        behavior = u_data[perm]  # behavior-biased indices (analog of PSM/FB z-mixing)
        mask = (jax.random.uniform(r_mix, (B,)) < c["index_mix_ratio"])[:, None]
        u_index = jnp.where(mask, behavior, gauss)
        u_index = jnp.clip(u_index, -c["u_clip"], c["u_clip"])
        return StepInputs(u_data=u_data, u_index=u_index)

    def apply_update(self, batch, sampled):
        tau = self.config["tau"]
        (_, info), (g_phi, g_psi) = jax.value_and_grad(self.measure_loss, argnums=(2, 3), has_aux=True)(
            batch, sampled, self.phi.params, self.psi.params)
        phi = self.phi.apply_gradients(grads=g_phi)
        psi = self.psi.apply_gradients(grads=g_psi)
        target_phi = polyak_update(phi.params, self.target_phi, tau)
        target_psi = polyak_update(psi.params, self.target_psi, tau)
        return self.replace(phi=phi, psi=psi, target_phi=target_phi, target_psi=target_psi), info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        sampled = self.sample_step_inputs(batch, rng)
        new_agent, info = self.apply_update(batch, sampled)
        return new_agent.replace(rng=new_rng), info

    def total_loss(self, batch, grad_params=None, rng=None):
        """Validation-logging loss at current params (no step)."""
        rng = rng if rng is not None else self.rng
        sampled = self.sample_step_inputs(batch, rng)
        loss, info = self.measure_loss(batch, sampled, self.phi.params, self.psi.params)
        return loss, info

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, rphi, rpsi, rvf, ronestep = jax.random.split(rng, 5)
        assert config.get("encoder", None) is None, "psmflow does not support visual encoders yet."
        action_dim = ex_actions.shape[-1]
        z_dim = config["z_dim"]
        ex_u = jnp.zeros((ex_observations.shape[0], action_dim))

        phi_def = PhiMap(z_dim=z_dim, hidden_dim=config["phi"]["hidden_dim"],
                         hidden_layers=config["phi"]["hidden_layers"], norm=True)
        psi_def = PsiMap(output_dim=z_dim, hidden_dim=config["sf"]["hidden_dim"],
                         num_parallel=config["num_parallel"],
                         embedding_layers=config["sf"]["embedding_layers"],
                         hidden_layers=config["sf"]["hidden_layers"])
        phi = TrainState.create(phi_def, phi_def.init(rphi, ex_observations)["params"],
                                tx=optax.adam(config["lr_phi"]))
        psi = TrainState.create(psi_def, psi_def.init(rpsi, ex_observations, ex_u, ex_u)["params"],
                                tx=optax.adam(config["lr_sf"]))

        # FROZEN behavior flow (FQL Stage-A checkpoint). Defs are rebuilt from config;
        # shapes must match the checkpointed run.
        flow_hidden = tuple(config["flow"]["hidden_dims"])
        vf_def = ActorVectorField(hidden_dims=flow_hidden, action_dim=action_dim,
                                  layer_norm=config["flow"]["layer_norm"])
        onestep_def = ActorVectorField(hidden_dims=flow_hidden, action_dim=action_dim,
                                       layer_norm=config["flow"]["layer_norm"])
        ckpt = config.get("flow_ckpt_path", None)
        if ckpt:
            flow_vf, flow_onestep = _load_flow_params(
                ckpt, config.get("flow_ckpt_epoch", None), config,
                ex_observations, ex_actions)
        else:
            assert config.get("allow_untrained_flow", False), (
                "psmflow requires agent.flow_ckpt_path (a Stage-A fql bc_only ckpt dir); "
                "set agent.allow_untrained_flow=true only for tests/smokes.")
            ex_times = ex_actions[..., :1]
            flow_vf = vf_def.init(rvf, ex_observations, ex_actions, ex_times)["params"]
            flow_onestep = onestep_def.init(ronestep, ex_observations, ex_actions)["params"]

        config = _plain_config(config)
        config["ob_dims"] = tuple(ex_observations.shape[1:])
        config["action_dim"] = action_dim
        return cls(rng=rng, phi=phi, psi=psi,
                   target_phi=copy.deepcopy(phi.params), target_psi=copy.deepcopy(psi.params),
                   flow_vf=flow_vf, flow_onestep=flow_onestep,
                   task_z=jnp.zeros((z_dim,), jnp.float32),
                   config=flax.core.FrozenDict(config),
                   flow_vf_def=vf_def, flow_onestep_def=onestep_def)


def _load_flow_params(ckpt_path, ckpt_epoch, config, ex_observations, ex_actions):
    """Extract the frozen flow subtrees from a Stage-A FQL(bc_only) checkpoint.

    Builds a throwaway FQLAgent with matching shapes and restores the pickle, then
    pulls modules_actor_bc_flow / modules_actor_onestep_flow.
    """
    from agents.fql import FQLAgent, get_config as fql_get_config
    from utils.flax_utils import restore_agent

    fql_cfg = fql_get_config()
    fql_cfg["actor_hidden_dims"] = tuple(config["flow"]["hidden_dims"])
    fql_cfg["value_hidden_dims"] = tuple(config["flow"]["value_hidden_dims"])
    fql_cfg["actor_layer_norm"] = config["flow"]["layer_norm"]
    fql_cfg["layer_norm"] = config["flow"]["critic_layer_norm"]
    fql_agent = FQLAgent.create(0, ex_observations, ex_actions, fql_cfg)
    fql_agent = restore_agent(fql_agent, ckpt_path, ckpt_epoch)
    params = fql_agent.network.params
    vf, onestep = params["modules_actor_bc_flow"], params["modules_actor_onestep_flow"]

    # restore_agent replaces the param tree wholesale and does NOT check shapes, so a
    # checkpoint from a different environment loads without complaint and only misbehaves
    # much later, at decode time. The bc-flow trunk takes concat[obs, action, t], so its
    # first kernel pins the env this flow was trained on.
    ob_dim = int(ex_observations.shape[-1])
    action_dim = int(ex_actions.shape[-1])
    got = int(vf["mlp"]["Dense_0"]["kernel"].shape[0])
    expected = ob_dim + action_dim + 1
    assert got == expected, (
        f'flow checkpoint {ckpt_path} expects concat[obs, action, t] of width {got}, but '
        f'this env gives {expected} (obs {ob_dim} + action {action_dim} + t 1); '
        'the Stage-A checkpoint was trained on a different environment')
    return vf, onestep


def get_config():
    """Importable default config, mirrored by configs/agent/psmflow.yaml."""
    import ml_collections

    return ml_collections.ConfigDict(
        dict(
            agent_name="psmflow",
            batch_size=1024,
            z_dim=128,
            num_parallel=2,
            discount=0.98,
            tau=0.01,
            ortho_coef=1000.0,       # reference PSM sweep winner
            pessimism_penalty=0.0,
            actor_pessimism_penalty=0.5,
            norm_z=True,
            lr_phi=1.0e-5,
            lr_sf=1.0e-4,
            phi=dict(hidden_dim=256, hidden_layers=2),
            sf=dict(hidden_dim=1024, hidden_layers=1, embedding_layers=2),
            # frozen behavior flow (must match the Stage-A fql bc_only run)
            flow=dict(hidden_dims=(512, 512, 512, 512), value_hidden_dims=(512, 512, 512, 512),
                      layer_norm=False, critic_layer_norm=True),
            flow_ckpt_path=ml_collections.config_dict.placeholder(str),
            flow_ckpt_epoch=ml_collections.config_dict.placeholder(int),
            allow_untrained_flow=False,
            preimage_path=ml_collections.config_dict.placeholder(str),
            use_point_preimage=False,
            # latent-family knobs
            index_mix_ratio=0.5,     # P(u' from behavior preimages) vs N(0, I)
            u_clip=3.0,              # typical-set clamp on all latent draws
            # flow-GPI inference
            gpi_num_u=64,
            gpi_num_uprime=16,
            gpi_decode="onestep",    # onestep | ode
            flow_decode_steps=10,    # Euler steps for gpi_decode=ode
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
