"""tests/test_psm_reference_equivalence.py — CORRECTNESS ORACLE (Task 4.1).

Proves our PSM (TD3-actor) nets + update math produce *bit-identical* outputs to
the reference (/home/adityamohan/git/baselines/PSM) given identical inputs and
weights: `torch.equal` (zero ULP), NOT `allclose`.

Reference import path
---------------------
The reference `models/psm_models.py` does `from ..utils import TruncatedNormal`,
a relative import requiring the reference to be imported as a *package*. We add
`/home/adityamohan/git/baselines` to sys.path and import `PSM.models.psm_models`
(the `PSM/` dir has an `__init__.py`). `PSM/utils.py` only pulls stdlib + numpy +
torch (no torchrl/dmc), so this imports cleanly — we use the REAL reference net
classes, no fallback needed. If this ever fails (e.g. heavy dep creep), the
import-based forward tests `pytest.importorskip`-skip; the loss/grad/param tests
build the reference modules from the same imported classes and skip too.

Everything runs on CPU, float64, deterministic algorithms.

Loss math transcribed VERBATIM from baselines/PSM/agent/psm.py:
  _update_psm  -> psm.py:438-477   (next_action from sampling_actor; INJECTED here)
  _update_sf   -> psm.py:495-532   (next_action from learned actor; INJECTED here)
We inject `next_action` via the default-None hook on PSMAgent._update_psm/_update_sf
to bypass the stochastic samplers, so both sides see identical inputs.
"""
import sys

import numpy as np
import pytest
import torch

REF = "/home/adityamohan/git/baselines"
if REF not in sys.path:
    sys.path.insert(0, REF)

torch.use_deterministic_algorithms(True)
DT = torch.float64
DEV = "cpu"


def _import_ref():
    return pytest.importorskip(
        "PSM.models.psm_models",
        reason="reference PSM.models.psm_models not importable under this torch env",
    )


def _copy(ref, ours):
    """Load ref weights into ours; load_state_dict succeeding proves identical
    module structure. If it raises, our vendoring diverged — let it propagate."""
    ours.load_state_dict(ref.state_dict())
    return ours


# ───────────────────────── Step B: forward bit-exactness ───────────────────── #

def test_phi_forward_bit_exact():
    refmod = _import_ref()
    RefPhi = refmod.PhiMap
    from agents.psm.psm_nets import PhiMap as OurPhi

    ref = RefPhi(40, 128, 256, 2, True, False).to(DEV).double().eval()
    ours = _copy(ref, OurPhi(40, 128, 256, 2, True, False).to(DEV).double().eval())
    x = torch.randn(8, 40, dtype=DT)
    assert torch.equal(ref(x), ours(x))


def test_psi_sf_forward_bit_exact():
    """SF-style PsiMap: z_dim=128 input, output_dim=z_dim (default)."""
    refmod = _import_ref()
    RefPsi = refmod.PsiMap
    from agents.psm.psm_nets import PsiMap as OurPsi

    obs_dim, z_dim, act_dim, hid, npar = 40, 128, 5, 1024, 2
    ref = RefPsi(obs_dim, z_dim, act_dim, hid, 1, 2, npar).to(DEV).double().eval()
    ours = _copy(ref, OurPsi(obs_dim, z_dim, act_dim, hid, 1, 2, npar).to(DEV).double().eval())
    obs = torch.randn(8, obs_dim, dtype=DT)
    z = torch.randn(8, z_dim, dtype=DT)
    action = torch.rand(8, act_dim, dtype=DT) * 2 - 1
    assert torch.equal(ref(obs, z, action), ours(obs, z, action))


def test_psi_psm_forward_bit_exact():
    """PSM-style PsiMap: z input dim = max_log_seed (16), output_dim=128."""
    refmod = _import_ref()
    RefPsi = refmod.PsiMap
    from agents.psm.psm_nets import PsiMap as OurPsi

    obs_dim, z_in, out_dim, act_dim, hid, npar = 40, 16, 128, 5, 1024, 2
    ref = RefPsi(obs_dim, z_in, act_dim, hid, 1, 2, npar, output_dim=out_dim).to(DEV).double().eval()
    ours = _copy(ref, OurPsi(obs_dim, z_in, act_dim, hid, 1, 2, npar, output_dim=out_dim).to(DEV).double().eval())
    obs = torch.randn(8, obs_dim, dtype=DT)
    z = torch.rand(8, z_in, dtype=DT).round()  # binary
    action = torch.rand(8, act_dim, dtype=DT) * 2 - 1
    assert torch.equal(ref(obs, z, action), ours(obs, z, action))


def test_actor_forward_bit_exact():
    """Compare TruncatedNormal dist.loc / dist.scale / dist.mean (RNG-free) and the
    clipped sample under a fixed manual_seed on each side."""
    refmod = _import_ref()
    RefActor = refmod.Actor
    from agents.psm.psm_nets import Actor as OurActor

    obs_dim, z_dim, act_dim, hid = 40, 128, 5, 1024
    ref = RefActor(obs_dim, z_dim, act_dim, hid, 1, 2).to(DEV).double().eval()
    ours = _copy(ref, OurActor(obs_dim, z_dim, act_dim, hid, 1, 2).to(DEV).double().eval())
    obs = torch.randn(8, obs_dim, dtype=DT)
    z = torch.randn(8, z_dim, dtype=DT)
    std = 0.2

    rd = ref(obs, z, std)
    od = ours(obs, z, std)
    assert torch.equal(rd.loc, od.loc)
    assert torch.equal(rd.scale, od.scale)
    assert torch.equal(rd.mean, od.mean)

    torch.manual_seed(0)
    rs = rd.sample(clip=0.3)
    torch.manual_seed(0)
    os_ = od.sample(clip=0.3)
    assert torch.equal(rs, os_)


# ─────────── shared dims/inputs for Step C (loss + grad + param) ───────────── #

OBS_DIM, ACT_DIM, Z_DIM, MAX_LOG_SEED, HID, NPAR, B = 40, 5, 128, 6, 1024, 2, 8
LR, BETAS, EPS, WD = 1e-4, (0.9, 0.999), 1e-8, 0.0


def _build_agent():
    import gymnasium as gym
    from agents.psm.agent import PSMAgent

    a = PSMAgent(
        obs_space=gym.spaces.Box(-1, 1, (OBS_DIM,)), action_dim=ACT_DIM, batch_size=B,
        z_dim=Z_DIM, max_log_seed=MAX_LOG_SEED, num_parallel=NPAR, device="cpu",
        lr_sf=LR, lr_phi=LR, lr_actor=LR, weight_decay=WD,
    )
    a.model.double()
    # off_diag / off_diag_sum were built float32 in setup_training; rebuild as float64
    a.off_diag = (1 - torch.eye(B, B, dtype=DT, device="cpu"))
    a.off_diag_sum = a.off_diag.sum()
    return a


def _fixed_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(B, OBS_DIM, generator=g, dtype=DT)
    action = (torch.rand(B, ACT_DIM, generator=g, dtype=DT) * 2 - 1)
    next_obs = torch.randn(B, OBS_DIM, generator=g, dtype=DT)
    goal = torch.randn(B, OBS_DIM, generator=g, dtype=DT)
    next_action = (torch.rand(B, ACT_DIM, generator=g, dtype=DT) * 2 - 1)
    # z_psm: binary, max_log_seed wide; z_sf: gaussian, z_dim wide
    z_psm = torch.randint(0, 2, (B, MAX_LOG_SEED), generator=g).to(DT)
    z_sf = torch.randn(B, Z_DIM, generator=g, dtype=DT)
    discount = torch.full((B, 1), 0.98, dtype=DT)
    return obs, action, next_obs, goal, next_action, z_psm, z_sf, discount


# ─────── reference loss transcriptions (psm.py:450-475 / 505-530) ──────────── #

def _get_targets_uncertainty(preds, num_parallel_scaling, dim=0):
    preds_mean = preds.mean(dim=dim)
    d1 = preds.unsqueeze(dim=dim)
    d2 = preds.unsqueeze(dim=dim + 1)
    preds_unc = torch.abs(d1 - d2).sum(dim=(dim, dim + 1)) / num_parallel_scaling
    return preds_mean, preds_unc


def _ref_psm_update(phi, psm_psi, target_phi, target_psm_psi, off_diag, off_diag_sum,
                    nps, opt, obs, action, discount, next_obs, goal, z, next_action,
                    pessimism_penalty=0.0):
    """VERBATIM transcription of reference _update_psm (psm.py:438-475)."""
    with torch.no_grad():
        target_psm_psis = target_psm_psi(next_obs, z, next_action)
        target_phi_o = target_phi(goal)
        target_Ms = torch.matmul(target_psm_psis, target_phi_o.T)
        target_M_mean, target_M_unc = _get_targets_uncertainty(target_Ms, nps)
        target_M = target_M_mean - pessimism_penalty * target_M_unc

    psis = psm_psi(obs, z, action)
    phi_o = phi(goal)
    Ms = torch.matmul(psis, phi_o.T)

    diff = Ms - discount * target_M
    psm_offdiag = 0.5 * (diff * off_diag).pow(2).sum() / off_diag_sum
    psm_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
    psm_loss = psm_offdiag + psm_diag

    Cov = torch.matmul(phi_o, phi_o.T)
    orth_loss_diag = -Cov.diag().mean()
    orth_loss_offdiag = 0.5 * (Cov * off_diag).pow(2).sum() / off_diag_sum
    orth_loss = orth_loss_offdiag + orth_loss_diag
    psm_loss = psm_loss + 1 * orth_loss

    opt.zero_grad(set_to_none=True)
    psm_loss.backward()
    return dict(psm_loss=psm_loss, psm_diag=psm_diag, psm_offdiag=psm_offdiag,
                orth_loss=orth_loss)


def _ref_sf_update(phi, sf_psi, target_sf_psi, actor, off_diag, off_diag_sum,
                   nps, ortho_coef, opt, obs, action, discount, next_obs, goal, z,
                   next_action, pessimism_penalty=0.0):
    """VERBATIM transcription of reference _update_sf (psm.py:495-530).
    Note: target_phi uses ONLINE self.phi (not target_phi); online phi is detached."""
    with torch.no_grad():
        target_psis = target_sf_psi(next_obs, z, next_action)
        target_phi_o = phi(goal)  # online phi, not target (ref line 499)
        target_Ms = torch.matmul(target_psis, target_phi_o.T)
        target_M_mean, target_M_unc = _get_targets_uncertainty(target_Ms, nps)
        target_M = target_M_mean - pessimism_penalty * target_M_unc

    psis = sf_psi(obs, z, action)
    phi_o = phi(goal).detach()  # detached (ref line 506)
    Ms = torch.matmul(psis, phi_o.T)

    diff = Ms - discount * target_M
    sf_offdiag = 0.5 * (diff * off_diag).pow(2).sum() / off_diag_sum
    sf_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
    sf_loss = sf_offdiag + sf_diag

    Cov = torch.matmul(phi_o, phi_o.T)
    orth_loss_diag = -Cov.diag().mean()
    orth_loss_offdiag = 0.5 * (Cov * off_diag).pow(2).sum() / off_diag_sum
    orth_loss = orth_loss_offdiag + orth_loss_diag
    sf_loss = sf_loss + 0 * ortho_coef * orth_loss

    opt.zero_grad(set_to_none=True)
    sf_loss.backward()
    return dict(sf_loss=sf_loss, sf_diag=sf_diag, sf_offdiag=sf_offdiag)


def _build_ref_nets():
    """Build reference nets + target copies (float64), structurally identical to ours."""
    import copy
    refmod = _import_ref()
    RefPhi, RefPsi = refmod.PhiMap, refmod.PsiMap
    phi = RefPhi(OBS_DIM, Z_DIM, 256, 2, True, False).to(DEV).double()
    sf_psi = RefPsi(OBS_DIM, Z_DIM, ACT_DIM, HID, 1, 2, NPAR).to(DEV).double()
    psm_psi = RefPsi(OBS_DIM, MAX_LOG_SEED, ACT_DIM, HID, 1, 2, NPAR, output_dim=Z_DIM).to(DEV).double()
    target_phi = copy.deepcopy(phi)
    target_sf_psi = copy.deepcopy(sf_psi)
    target_psm_psi = copy.deepcopy(psm_psi)
    return phi, sf_psi, psm_psi, target_phi, target_sf_psi, target_psm_psi


def _sync_ours_from_ref(agent, phi, sf_psi, psm_psi, target_phi, target_sf_psi, target_psm_psi):
    """Copy ref weights into our model's nets + targets so both sides start identical."""
    agent.model.phi.load_state_dict(phi.state_dict())
    agent.model.sf_psi.load_state_dict(sf_psi.state_dict())
    agent.model.psm_psi.load_state_dict(psm_psi.state_dict())
    agent.model.target_phi.load_state_dict(target_phi.state_dict())
    agent.model.target_sf_psi.load_state_dict(target_sf_psi.state_dict())
    agent.model.target_psm_psi.load_state_dict(target_psm_psi.state_dict())


def _assert_grads_equal(ref_params, our_params, label):
    rp = list(ref_params)
    op = list(our_params)
    assert len(rp) == len(op), f"{label}: param count mismatch {len(rp)} vs {len(op)}"
    for i, (r, o) in enumerate(zip(rp, op)):
        assert (r.grad is None) == (o.grad is None), f"{label}: grad-None mismatch at {i}"
        if r.grad is not None:
            assert torch.equal(r.grad, o.grad), f"{label}: grad mismatch at param {i}"


def _assert_params_equal(ref_params, our_params, label):
    for i, (r, o) in enumerate(zip(list(ref_params), list(our_params))):
        assert torch.equal(r, o), f"{label}: param mismatch at {i}"


# ─────────────────── Step C: psm update bit-exactness ──────────────────────── #

def test_psm_update_bit_exact():
    _import_ref()
    agent = _build_agent()
    obs, action, next_obs, goal, next_action, z_psm, z_sf, discount = _fixed_inputs()
    phi, sf_psi, psm_psi, target_phi, target_sf_psi, target_psm_psi = _build_ref_nets()
    _sync_ours_from_ref(agent, phi, sf_psi, psm_psi, target_phi, target_sf_psi, target_psm_psi)
    nps = NPAR ** 2 - NPAR

    # Fresh, identical Adam optimizers on each side (no pre-warmed state).
    ref_opt = torch.optim.Adam(list(phi.parameters()) + list(psm_psi.parameters()),
                               lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
    agent.optim_phi = torch.optim.Adam(agent.model.phi.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
    agent.optim_psm_psi = torch.optim.Adam(agent.model.psm_psi.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)

    # Our `_update_psm` does zero_grad + backward + optim.step() INTERNALLY (one
    # full Adam step). The ref helper does zero_grad + backward only; we step its
    # optimizer ONCE below so both sides take exactly one step. (next_obs_hash is
    # unused when next_action is injected.)
    ours = agent._update_psm(obs, action, discount, next_obs,
                             torch.arange(B), goal, z_psm, next_action=next_action)
    ref = _ref_psm_update(phi, psm_psi, target_phi, target_psm_psi, agent.off_diag,
                          agent.off_diag_sum, nps, ref_opt, obs, action, discount,
                          next_obs, goal, z_psm, next_action,
                          pessimism_penalty=agent.pessimism_penalty)

    # loss-term bit-exactness
    assert torch.equal(ref["psm_loss"], ours["psm_loss"]), "psm_loss"
    assert torch.equal(ref["psm_diag"], ours["psm_diag"]), "psm_diag"
    assert torch.equal(ref["psm_offdiag"], ours["psm_offdiag"]), "psm_offdiag"
    assert torch.equal(ref["orth_loss"], ours["orth_loss"]), "orth_loss"

    # gradient bit-exactness (phi + psm_psi). Both sides have stepped 0 (ref) / 1
    # (ours) times, but Adam.step does not clear .grad, so the grads still match.
    _assert_grads_equal(phi.parameters(), agent.model.phi.parameters(), "psm:phi.grad")
    _assert_grads_equal(psm_psi.parameters(), agent.model.psm_psi.parameters(), "psm:psm_psi.grad")

    # step the ref optimizer ONCE (ours already stepped inside _update_psm), then
    # compare params bit-exact.
    ref_opt.step()
    _assert_params_equal(phi.parameters(), agent.model.phi.parameters(), "psm:phi.param")
    _assert_params_equal(psm_psi.parameters(), agent.model.psm_psi.parameters(), "psm:psm_psi.param")


# ─────────────────── Step C: sf update bit-exactness ───────────────────────── #

def test_sf_update_bit_exact():
    _import_ref()
    agent = _build_agent()
    obs, action, next_obs, goal, next_action, z_psm, z_sf, discount = _fixed_inputs()
    phi, sf_psi, psm_psi, target_phi, target_sf_psi, target_psm_psi = _build_ref_nets()
    _sync_ours_from_ref(agent, phi, sf_psi, psm_psi, target_phi, target_sf_psi, target_psm_psi)
    nps = NPAR ** 2 - NPAR

    ref_opt = torch.optim.Adam(sf_psi.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
    agent.optim_sf_psi = torch.optim.Adam(agent.model.sf_psi.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)

    # Our `_update_sf` does zero_grad + backward + optim.step() INTERNALLY.
    ours = agent._update_sf(obs, action, discount, next_obs, goal, z_sf,
                            next_action=next_action)
    ref = _ref_sf_update(phi, sf_psi, target_sf_psi, None, agent.off_diag,
                         agent.off_diag_sum, nps, agent.ortho_coef, ref_opt, obs,
                         action, discount, next_obs, goal, z_sf, next_action,
                         pessimism_penalty=agent.pessimism_penalty)

    assert torch.equal(ref["sf_loss"], ours["sf_loss"]), "sf_loss"
    assert torch.equal(ref["sf_diag"], ours["sf_diag"]), "sf_diag"
    assert torch.equal(ref["sf_offdiag"], ours["sf_offdiag"]), "sf_offdiag"

    _assert_grads_equal(sf_psi.parameters(), agent.model.sf_psi.parameters(), "sf:sf_psi.grad")

    # step the ref optimizer ONCE (ours already stepped inside _update_sf).
    ref_opt.step()
    _assert_params_equal(sf_psi.parameters(), agent.model.sf_psi.parameters(), "sf:sf_psi.param")


# ─────────────────── Step C: actor update bit-exactness ────────────────────── #

def _ref_actor_update(actor, sf_psi, nps, actor_pessimism_penalty, opt,
                      obs, z, action):
    """VERBATIM transcription of reference _update_td3_actor (psm.py:651-665).
    `action` is the (already-sampled) injected action routed through the actor
    dist's straight-through estimator so gradients flow to the actor exactly as
    in the reference `dist.sample(clip)` path (TruncatedNormal._clamp)."""
    dist = actor(obs, z, 0.2)                          # stddev unused once action injected
    action = dist._clamp(dist.loc + (action - dist.loc).detach())
    psis = sf_psi(obs, z, action)                      # P x B x z_dim
    Qs = (psis * z).sum(-1)                             # P x B
    Q_mean, Q_unc = _get_targets_uncertainty(Qs, nps)   # B
    Q = Q_mean - actor_pessimism_penalty * Q_unc        # B
    actor_loss = -Q.mean()

    opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    return dict(actor_loss=actor_loss, q=Q.mean(), Q=Q)


def test_actor_update_bit_exact():
    refmod = _import_ref()
    RefActor, RefPsi = refmod.Actor, refmod.PsiMap
    agent = _build_agent()
    obs, action_unused, next_obs, goal, next_action, z_psm, z_sf, discount = _fixed_inputs()
    nps = NPAR ** 2 - NPAR
    app = agent.actor_pessimism_penalty

    # Build ref actor + sf_psi (float64, CPU) structurally identical to ours and
    # copy ref weights into ours so both sides start from identical parameters.
    ref_actor = RefActor(OBS_DIM, Z_DIM, ACT_DIM, HID, 1, 2).to(DEV).double()
    ref_sf_psi = RefPsi(OBS_DIM, Z_DIM, ACT_DIM, HID, 1, 2, NPAR).to(DEV).double()
    agent.model.actor.load_state_dict(ref_actor.state_dict())
    agent.model.sf_psi.load_state_dict(ref_sf_psi.state_dict())

    # Fixed injected action: sampled ONCE from the ref dist (clipped), then detached.
    # Both sides receive this exact tensor, so no dist.sample RNG enters the compare.
    with torch.no_grad():
        a_inj = ref_actor(obs, z_sf, agent.actor_std).sample(clip=agent.stddev_clip).detach()

    # Fresh, identical Adam optimizers on each side (no pre-warmed state).
    ref_opt = torch.optim.Adam(ref_actor.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
    agent.optim_actor = torch.optim.Adam(agent.model.actor.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)

    # Our `_update_actor` does zero_grad + backward + optim.step() INTERNALLY.
    ours = agent._update_actor(obs, z_sf, action=a_inj)
    ref = _ref_actor_update(ref_actor, ref_sf_psi, nps, app, ref_opt, obs, z_sf, a_inj)

    # loss + Q bit-exactness
    assert torch.equal(ref["actor_loss"], ours["actor_loss"]), "actor_loss"
    assert torch.equal(ref["q"], ours["q"]), "q"

    # gradient bit-exactness (actor only — sf_psi is upstream but not optimized here).
    _assert_grads_equal(ref_actor.parameters(), agent.model.actor.parameters(), "actor:actor.grad")

    # step the ref optimizer ONCE (ours already stepped inside _update_actor).
    ref_opt.step()
    _assert_params_equal(ref_actor.parameters(), agent.model.actor.parameters(), "actor:actor.param")
