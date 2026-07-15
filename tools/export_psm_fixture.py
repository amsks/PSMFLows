"""tools/export_psm_fixture.py — run in the Factored-FB PyTorch venv.

Exports a numerical fixture of the reference PSM (small dims, float64) so the JAX
port can be checked for equivalence WITHOUT importing torch.

Usage:
  /Users/adityamohan/git/Austin/FB-Manipulation/Factored-FB/.venv/bin/python \
    tools/export_psm_fixture.py --out tests/fixtures/psm_reference.npz

The fixture contains, as named arrays:
  w__<torch_state_dict_key>      : every reference module parameter (online + target)
  proto_seed_to_action, proto_powers
  in__<name>                     : fixed batch + ALL injected randomness
  out__<name>                    : reference forward outputs / losses (no optimizer step)
  grad__<group>__<param>         : reference grads of each stage loss (no step)
  step__<i>__<metric>            : per-step metrics from a K-step injected replay
  step_in__<i>__<name>           : per-step injected randomness
"""

import argparse
import sys

import numpy as np
import torch

# Import the reference (torch) PSM from the Factored-FB tree.
sys.path.insert(0, "/Users/adityamohan/git/Austin/FB-Manipulation/Factored-FB")
import gymnasium as gym  # noqa: E402
from agents.psm.agent import PSMAgent  # noqa: E402

D = dict(obs_dim=8, action_dim=2, batch=16, z_dim=8, hidden=32,
         num_parallel=2, max_log_seed=4)
HP = dict(discount=0.98, tau=0.01, ortho_coef=1.0, mix_ratio=0.5,
          pessimism_penalty=0.0, actor_pessimism_penalty=0.5,
          actor_std=0.2, stddev_clip=0.3, lr=1e-4)
K_STEPS = 10


def make_agent():
    return PSMAgent(
        obs_space=gym.spaces.Box(-1, 1, (D["obs_dim"],)), action_dim=D["action_dim"],
        batch_size=D["batch"], z_dim=D["z_dim"], max_log_seed=D["max_log_seed"],
        phi_cfg={"hidden_dim": D["hidden"], "hidden_layers": 2, "norm": True, "batch_norm": False},
        sf_cfg={"hidden_dim": D["hidden"], "hidden_layers": 1, "embedding_layers": 2},
        actor_cfg={"hidden_dim": D["hidden"], "hidden_layers": 1, "embedding_layers": 2,
                   "std": HP["actor_std"], "stddev_clip": HP["stddev_clip"]},
        num_parallel=D["num_parallel"], discount=HP["discount"],
        ortho_coef=HP["ortho_coef"], mix_ratio=HP["mix_ratio"],
        pessimism_penalty=HP["pessimism_penalty"],
        actor_pessimism_penalty=HP["actor_pessimism_penalty"],
        actor_std=HP["actor_std"], stddev_clip=HP["stddev_clip"],
        lr_sf=HP["lr"], lr_phi=HP["lr"], lr_actor=HP["lr"],
        target_tau=HP["tau"], device="cpu", actor_kind="td3")


def npy(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


def main(out):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    agent = make_agent()
    m = agent.model
    B = D["batch"]
    off_diag = (1 - torch.eye(B))
    off_sum = off_diag.sum()

    fix = {}
    for k, v in m.state_dict().items():
        fix[f"w__{k}"] = npy(v)
    fix["proto_seed_to_action"] = np.asarray(m.proto_sampler.seed_to_action)
    fix["proto_powers"] = npy(m.proto_sampler.powers)

    # ---- fixed batch + injected randomness ----
    torch.manual_seed(1)
    O, A, Z = D["obs_dim"], D["action_dim"], D["z_dim"]
    obs = torch.rand(B, O) * 2 - 1
    action = torch.rand(B, A) * 2 - 1
    next_obs = torch.rand(B, O) * 2 - 1
    goal = m._normalize(next_obs)                       # Identity for state
    z_cont = m.project_z(torch.randn(B, Z))
    z_psm = m.sample_z_psm(B, device="cpu")
    next_obs_hash = torch.arange(B)
    proto_next_action = m.proto_sampler(next_obs_hash, z_psm)
    actor_next_action = torch.rand(B, A) * 2 - 1        # injected SF-branch next action
    actor_sample = torch.rand(B, A) * 2 - 1             # injected actor-update sample
    for n, t in dict(obs=obs, action=action, next_obs=next_obs, goal=goal,
                     z_cont=z_cont, z_psm=z_psm, next_obs_hash=next_obs_hash,
                     proto_next_action=proto_next_action,
                     actor_next_action=actor_next_action, actor_sample=actor_sample).items():
        fix[f"in__{n}"] = npy(t)

    disc = (HP["discount"] * torch.ones(B, 1))

    def uncert(preds):  # mirror model.get_targets_uncertainty (dim=0)
        mean = preds.mean(dim=0)
        d1 = preds.unsqueeze(0); d2 = preds.unsqueeze(1)
        unc = torch.abs(d1 - d2).sum(dim=(0, 1)) / m.num_parallel_scaling
        return mean, unc

    def contrastive(Ms, target_M):
        diff = Ms - disc * target_M
        offd = 0.5 * (diff * off_diag).pow(2).sum() / off_sum
        diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
        return offd + diag, diag, offd

    def ortho(phi):
        cov = torch.matmul(phi, phi.T)
        offd = 0.5 * (cov * off_diag).pow(2).sum() / off_sum
        diag = -cov.diag().mean()
        return offd + diag, diag, offd

    # ---- static PROTO outputs + grads (no step) ----
    with torch.no_grad():
        tgt_psm = m.target_psm_psi(next_obs, z_psm, proto_next_action)
        tgt_phi = m.target_phi(goal)
        tgt_Ms = torch.matmul(tgt_psm, tgt_phi.T)
        tmean, tunc = uncert(tgt_Ms)
        target_M_psm = tmean - HP["pessimism_penalty"] * tunc
    psis = m.psm_psi(obs, z_psm, action)
    phi = m.phi(goal)
    Ms = torch.matmul(psis, phi.T)
    psm_loss, psm_diag, psm_offdiag = contrastive(Ms, target_M_psm)
    o_loss, o_diag, o_offdiag = ortho(phi)
    psm_total = psm_loss + HP["ortho_coef"] * o_loss
    g = torch.autograd.grad(psm_total, list(m.phi.parameters()) + list(m.psm_psi.parameters()))
    n_phi = len(list(m.phi.parameters()))
    for (name, _), gv in zip(list(m.phi.named_parameters()) + list(m.psm_psi.named_parameters()), g):
        grp = "phi" if name in dict(m.phi.named_parameters()) else "psm_psi"
        fix[f"grad__{grp}__{name}"] = npy(gv)
    fix["out__phi"] = npy(phi)
    fix["out__psm_psi"] = npy(psis)
    fix["out__M_psm"] = npy(Ms)
    fix["out__target_M_psm"] = npy(target_M_psm)
    for n_, v_ in dict(psm_loss=psm_loss, psm_diag=psm_diag, psm_offdiag=psm_offdiag,
                       orth_loss=o_loss, orth_diag=o_diag, orth_offdiag=o_offdiag,
                       psm_total=psm_total).items():
        fix[f"out__{n_}"] = npy(v_)

    # ---- static SF outputs + grads (uses self.phi NOT target_phi; phi detached) ----
    with torch.no_grad():
        tgt_sf = m.target_sf_psi(next_obs, z_cont, actor_next_action)
        tgt_phi_sf = m.phi(goal)                         # NOTE: self.phi, not target
        tgt_Ms_sf = torch.matmul(tgt_sf, tgt_phi_sf.T)
        smean, sunc = uncert(tgt_Ms_sf)
        target_M_sf = smean - HP["pessimism_penalty"] * sunc
    sf_psis = m.sf_psi(obs, z_cont, action)
    phi_det = m.phi(goal).detach()
    Ms_sf = torch.matmul(sf_psis, phi_det.T)
    sf_loss, sf_diag, sf_offdiag = contrastive(Ms_sf, target_M_sf)
    g_sf = torch.autograd.grad(sf_loss, list(m.sf_psi.parameters()))
    for (name, _), gv in zip(m.sf_psi.named_parameters(), g_sf):
        fix[f"grad__sf_psi__{name}"] = npy(gv)
    fix["out__sf_psi"] = npy(sf_psis)
    fix["out__M_sf"] = npy(Ms_sf)
    fix["out__target_M_sf"] = npy(target_M_sf)
    for n_, v_ in dict(sf_loss=sf_loss, sf_diag=sf_diag, sf_offdiag=sf_offdiag).items():
        fix[f"out__{n_}"] = npy(v_)

    # ---- static ACTOR outputs + grads (inject sample via straight-through clamp) ----
    dist = m.actor(obs, z_cont, HP["actor_std"])
    a_st = dist._clamp(dist.loc + (actor_sample - dist.loc).detach())
    q_psis = m.sf_psi(obs, z_cont, a_st)
    Qs = (q_psis * z_cont).sum(-1)
    q_mean, q_unc = uncert(Qs)
    Q = q_mean - HP["actor_pessimism_penalty"] * q_unc
    actor_loss = -Q.mean()
    g_a = torch.autograd.grad(actor_loss, list(m.actor.parameters()))
    for (name, _), gv in zip(m.actor.named_parameters(), g_a):
        fix[f"grad__actor__{name}"] = npy(gv)
    fix["out__actor_mu"] = npy(dist.loc)
    fix["out__actor_loss"] = npy(actor_loss)
    fix["out__q"] = npy(Q.mean())

    # ---- K-step injected replay (fresh agent, same init) ----
    torch.manual_seed(0)
    agent2 = make_agent()
    m2 = agent2.model
    from nn_models import _soft_update_params
    torch.manual_seed(123)
    for i in range(K_STEPS):
        zp = m2.sample_z_psm(B, device="cpu")
        zc = m2.project_z(torch.randn(B, Z))
        pna = torch.rand(B, A) * 2 - 1
        ana = torch.rand(B, A) * 2 - 1
        asm = torch.rand(B, A) * 2 - 1
        for n_, t_ in dict(z_psm=zp, z_cont=zc, proto_next_action=pna,
                           actor_next_action=ana, actor_sample=asm).items():
            fix[f"step_in__{i}__{n_}"] = npy(t_)
        mi = {}
        mi.update(agent2._update_psm(obs, action, disc, next_obs, next_obs_hash, goal, zp, next_action=pna))
        with torch.no_grad():
            _soft_update_params(agent2._psm_psi_paramlist, agent2._target_psm_psi_paramlist, HP["tau"])
            _soft_update_params(agent2._phi_paramlist, agent2._target_phi_paramlist, HP["tau"])
        mi.update(agent2._update_sf(obs, action, disc, next_obs, goal, zc, next_action=ana))
        with torch.no_grad():
            _soft_update_params(agent2._sf_psi_paramlist, agent2._target_sf_psi_paramlist, HP["tau"])
        mi.update(agent2._update_actor(obs, zc, action=asm))
        for k_, v_ in mi.items():
            fix[f"step__{i}__{k_}"] = npy(v_)

    np.savez(out, **fix)
    print(f"wrote {out} with {len(fix)} arrays")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    main(ap.parse_args().out)
