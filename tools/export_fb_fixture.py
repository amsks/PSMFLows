"""tools/export_fb_fixture.py — run in the Factored-FB PyTorch venv.

Exports a numerical fixture of the reference FB (FlowBC) agent at small dims,
float64, so the JAX port can be checked for bit-exact equivalence WITHOUT torch.

Usage:
  /var/local/amsks/ffb-venv/bin/python tools/export_fb_fixture.py \
    --repo /u/amsks/git/Factored-FB --out tests/fixtures/fb_reference.npz

Mirrors tools/export_psm_fixture.py but for the cube-default FB flowbc path
(measure critic, perm goal mode, onestep off, penalties 0, q_loss off). The
update is reconstructed manually from the injected randomness so every draw is
recorded and the JAX side can inject the same values.

Namespaces in the .npz:
  w__<state_dict_key>        every model param incl. _target_* (online + target)
  in__<name>                 fixed batch + injected randomness for one update
  out__net_*                 clean per-module forward outputs (network equiv tests)
  out__<loss>                pre-optimizer-step scalar losses (agent static equiv)
  grad__<net>__<key>         grads of fb_loss / actor_loss at the fixed inputs
  step_in__<i>__<name>       per-step injected randomness (i=0..9)
  step__<i>__<state_dict_key> params after each full update (i=0..9)
"""

import argparse
import math
import sys

import numpy as np
import torch


D = dict(obs_dim=8, action_dim=5, batch=16, z_dim=8, L_dim=8, hidden=32, num_parallel=2)
HP = dict(discount=0.99, f_tau=0.005, b_tau=0.005, ortho_coef=1.0, train_goal_ratio=0.5,
          fb_pess=0.0, actor_pess=0.0, actor_std=0.2, stddev_clip=0.3, bc_coeff=3.0,
          flow_steps=10, lr_f=1e-4, lr_b=1e-4, lr_actor=1e-4, lr_actor_vf=3e-4)
K_STEPS = 10


def npy(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


def make_agent(gym, cfgmods):
    from agents.fb.flow_bc.agent import FBFlowBCAgent
    return FBFlowBCAgent(
        obs_space=gym.spaces.Box(-1, 1, (D["obs_dim"],)),
        action_dim=D["action_dim"],
        actor_cfg=cfgmods["NoiseConditionedActorArchiConfig"](
            hidden_dim=D["hidden"], hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=cfgmods["SimpleVectorFieldArchiConfig"](
            hidden_dim=D["hidden"], hidden_layers=4),
        flow_steps=HP["flow_steps"], lr_actor_vf=HP["lr_actor_vf"],
        batch_size=D["batch"], z_dim=D["z_dim"], L_dim=D["L_dim"], actor_encode_obs=False,
        forward_cfg=cfgmods["ForwardArchiConfig"](
            hidden_dim=D["hidden"], hidden_layers=2, embedding_layers=2,
            num_parallel=D["num_parallel"]),
        backward_cfg=cfgmods["BackwardArchiConfig"](
            hidden_dim=D["hidden"], hidden_layers=4, norm=True),
        left_encoder_cfg=cfgmods["BackwardArchiConfig"](
            hidden_dim=D["hidden"], hidden_layers=4, norm=True),
        discount=HP["discount"], lr_f=HP["lr_f"], lr_b=HP["lr_b"], lr_actor=HP["lr_actor"],
        weight_decay=0.0, clip_grad_norm=0.0, ortho_coef=HP["ortho_coef"],
        train_goal_ratio=HP["train_goal_ratio"], fb_pessimism_penalty=HP["fb_pess"],
        actor_pessimism_penalty=HP["actor_pess"], actor_std=HP["actor_std"],
        stddev_clip=HP["stddev_clip"], f_target_tau=HP["f_tau"], b_target_tau=HP["b_tau"],
        bc_coeff=HP["bc_coeff"], q_loss_coef=0.0, onestep=False, goal_cond=False,
        fixed_b="none", device="cpu")


def main(repo, out):
    sys.path.insert(0, repo)
    import gymnasium as gym
    import nn_models as NM
    from nn_models import _soft_update_params

    cfgmods = {n: getattr(NM, n) for n in
               ["ForwardArchiConfig", "BackwardArchiConfig",
                "NoiseConditionedActorArchiConfig", "SimpleVectorFieldArchiConfig"]}

    torch.set_default_dtype(torch.float64)
    B, O, A, Z = D["batch"], D["obs_dim"], D["action_dim"], D["z_dim"]
    P = D["num_parallel"]
    off_diag = 1 - torch.eye(B)
    off_sum = off_diag.sum()

    def uncert(preds, penalty):  # get_targets_uncertainty (dim=0)
        mean = preds.mean(dim=0)
        d = torch.abs(preds.unsqueeze(0) - preds.unsqueeze(1)).sum(dim=(0, 1))
        scale = preds.shape[0] ** 2 - preds.shape[0]
        unc = d / scale
        return mean, unc, mean - penalty * unc

    def project_z(z):
        return math.sqrt(z.shape[-1]) * torch.nn.functional.normalize(z, dim=-1)

    def flow_actions(m, obs, noises):
        actions = noises
        for i in range(HP["flow_steps"]):
            t = torch.ones((noises.shape[0], 1)) * i / HP["flow_steps"]
            actions = actions + m._actor_vf(obs, actions, t) / HP["flow_steps"]
        return torch.clamp(actions, -1, 1)

    def draw(seed):
        torch.manual_seed(seed)
        return dict(
            z_gauss=project_z(torch.randn(B, Z)),
            mix_mask=(torch.rand(B, 1) < HP["train_goal_ratio"]),
            perm=torch.randperm(B),
            next_actor_noise=torch.randn(B, A),
            flow_x0=torch.randn(B, A),
            flow_t=torch.rand(B, 1),
            actor_noise=torch.randn(B, A),
        )

    def mixed_z(m, inj, next_obs):
        with torch.no_grad():  # sample_mixed_z is @torch.no_grad() in the reference
            goals = project_z(m._backward_map(next_obs[inj["perm"]]))
            return torch.where(inj["mix_mask"], goals, inj["z_gauss"])

    def fb_step(m, agent, obs, action, next_obs, goal, disc, z, next_action, do_step):
        with torch.no_grad():
            nle = m._target_left_encoder(next_obs)
            tFs = m._target_forward_map(nle, z, next_action)
            tB = m._target_backward_map(goal)
            tMs = torch.matmul(tFs, tB.T)
            _, _, tM = uncert(tMs, HP["fb_pess"])
        left_enc = m._left_encoder(obs)
        Fs = m._forward_map(left_enc, z, action)
        Bm = m._backward_map(goal)
        Ms = torch.matmul(Fs, Bm.T)
        diff = Ms - disc * tM
        fb_off = 0.5 * (diff * off_diag).pow(2).sum() / off_sum
        fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * P
        Cov = torch.matmul(Bm, Bm.T)
        orth = 0.5 * (Cov * off_diag).pow(2).sum() / off_sum - Cov.diag().mean()
        fb_loss = fb_off + fb_diag + HP["ortho_coef"] * orth
        out = dict(fb_loss=fb_loss, fb_offdiag=fb_off, fb_diag=fb_diag, ortho_loss=orth)
        if do_step:
            agent.forward_optimizer.zero_grad(set_to_none=True)
            agent.backward_optimizer.zero_grad(set_to_none=True)
            fb_loss.backward()
            agent.forward_optimizer.step()
            agent.backward_optimizer.step()
        return out

    def actor_step(m, agent, obs, action, z, x0, t, noises, do_step):
        x1 = action
        xt = (1 - t) * x0 + t * x1
        vel = x1 - x0
        pred = m._actor_vf(obs, xt, t)
        bc_flow = (pred - vel).pow(2).mean()
        with torch.no_grad():
            left_enc = m._left_encoder(obs)
        aa = m._actor(obs, z, noises)
        Fs = m._forward_map(left_enc, z, aa)
        Qs = (Fs * z).sum(-1)
        _, _, Q = uncert(Qs, HP["actor_pess"])
        actor_loss = -Q.mean()
        with torch.no_grad():
            tfa = flow_actions(m, obs, noises)
        bc_err = (aa - tfa).pow(2).mean()
        actor_loss = actor_loss / Qs.abs().mean().detach() + HP["bc_coeff"] * bc_err + bc_flow
        out = dict(actor_loss=actor_loss, bc_flow_loss=bc_flow, bc_error=bc_err, q=Q.mean())
        if do_step:
            agent.actor_optimizer.zero_grad(set_to_none=True)
            agent.actor_vf_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            agent.actor_optimizer.step()
            agent.actor_vf_optimizer.step()
        return out

    def soft(agent):
        with torch.no_grad():
            _soft_update_params(agent._forward_map_paramlist, agent._target_forward_map_paramlist, HP["f_tau"])
            _soft_update_params(agent._backward_map_paramlist, agent._target_backward_map_paramlist, HP["b_tau"])
            _soft_update_params(agent._left_encoder_paramlist, agent._target_left_encoder_paramlist, HP["f_tau"])

    # ---------------- build agent + dump params ----------------
    torch.manual_seed(0)
    agent = make_agent(gym, cfgmods)
    m = agent.model
    fix = {}
    for k, v in m.state_dict().items():
        fix[f"w__{k}"] = npy(v)

    # ---------------- fixed batch + injected randomness ----------------
    torch.manual_seed(1)
    obs = torch.rand(B, O) * 2 - 1
    action = torch.rand(B, A) * 2 - 1
    next_obs = torch.rand(B, O) * 2 - 1
    rewards = torch.rand(B, 1)
    terminals = torch.zeros(B)
    goal = next_obs  # bw_encoder = Identity for state
    inj = draw(2)
    for n, t in dict(observations=obs, actions=action, next_observations=next_obs,
                     rewards=rewards, terminals=terminals, **inj).items():
        fix[f"in__{n}"] = npy(t)

    disc = (HP["discount"] * (1.0 - terminals)).reshape(-1, 1)
    z = mixed_z(m, inj, next_obs)
    with torch.no_grad():
        next_action = m._actor(next_obs, z, inj["next_actor_noise"])  # flowbc next-action
    fix["in__z"] = npy(z)                      # final mixed z fed to the loss
    fix["in__next_action"] = npy(next_action)  # stop-grad bootstrap action

    # ---------------- clean per-module outputs (network equiv) ----------------
    with torch.no_grad():
        left_enc = m._left_encoder(obs)
        fix["out__left_enc"] = npy(left_enc)
        fix["out__B"] = npy(m._backward_map(obs))
        fix["out__F"] = npy(m._forward_map(left_enc, inj["z_gauss"], action))
        fix["out__actor_mu"] = npy(m._actor(obs, inj["z_gauss"], inj["actor_noise"]))
        fix["out__vf"] = npy(m._actor_vf(obs, action, inj["flow_t"]))

    # ---------------- static losses + grads (no optimizer step) ----------------
    fb = fb_step(m, agent, obs, action, next_obs, goal, disc, z, next_action, do_step=False)
    g_fb = torch.autograd.grad(
        fb["fb_loss"],
        list(m._forward_map.parameters()) + list(m._left_encoder.parameters())
        + list(m._backward_map.parameters()), allow_unused=True)
    names = ([("forward", n) for n, _ in m._forward_map.named_parameters()]
             + [("left_encoder", n) for n, _ in m._left_encoder.named_parameters()]
             + [("backward", n) for n, _ in m._backward_map.named_parameters()])
    for (grp, name), gv in zip(names, g_fb):
        if gv is not None:
            fix[f"grad__{grp}__{name}"] = npy(gv)
    for k, v in fb.items():
        fix[f"out__{k}"] = npy(v)

    act = actor_step(m, agent, obs, action, z, inj["flow_x0"], inj["flow_t"],
                     inj["actor_noise"], do_step=False)
    g_a = torch.autograd.grad(
        act["actor_loss"],
        list(m._actor.parameters()) + list(m._actor_vf.parameters()), allow_unused=True)
    anames = ([("actor", n) for n, _ in m._actor.named_parameters()]
              + [("actor_vf", n) for n, _ in m._actor_vf.named_parameters()])
    for (grp, name), gv in zip(anames, g_a):
        if gv is not None:
            fix[f"grad__{grp}__{name}"] = npy(gv)
    for k, v in act.items():
        fix[f"out__{k}"] = npy(v)

    # ---------------- K-step injected replay (fresh agent, same init) ----------------
    torch.manual_seed(0)
    agent2 = make_agent(gym, cfgmods)
    m2 = agent2.model
    for i in range(K_STEPS):
        inj_i = draw(1000 + i)
        for n, t in inj_i.items():
            fix[f"step_in__{i}__{n}"] = npy(t)
        z_i = mixed_z(m2, inj_i, next_obs)
        with torch.no_grad():
            na_i = m2._actor(next_obs, z_i, inj_i["next_actor_noise"])
        fix[f"step_in__{i}__z"] = npy(z_i)
        fix[f"step_in__{i}__next_action"] = npy(na_i)
        fb_step(m2, agent2, obs, action, next_obs, goal, disc, z_i, na_i, do_step=True)
        actor_step(m2, agent2, obs.detach(), action, z_i, inj_i["flow_x0"],
                   inj_i["flow_t"], inj_i["actor_noise"], do_step=True)
        soft(agent2)
        for k, v in m2.state_dict().items():
            fix[f"step__{i}__{k}"] = npy(v)

    np.savez(out, **fix)
    print(f"wrote {out} with {len(fix)} arrays")

    # schema sanity
    need = ["in__observations", "in__z_gauss", "in__mix_mask", "in__perm",
            "in__z", "in__next_action", "in__flow_x0", "in__flow_t", "in__actor_noise",
            "out__F", "out__B", "out__left_enc", "out__fb_loss", "out__actor_loss",
            "out__actor_mu", "out__vf",
            "step_in__0__z", "step_in__0__next_action"]
    for k in need:
        assert k in fix, f"MISSING {k}"
    assert fix["out__F"].ndim == 3 and fix["out__F"].shape[0] == P
    print("schema ok; out__F shape", fix["out__F"].shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/u/amsks/git/Factored-FB")
    ap.add_argument("--out", default="tests/fixtures/fb_reference.npz")
    a = ap.parse_args()
    main(a.repo, a.out)
