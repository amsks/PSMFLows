import torch
from agents.tdmpc2.config import build_tdmpc2_cfg
from agents.tdmpc2.world_model import WorldModel


def _wm(obs_dim=12, act_dim=4):
    cfg = build_tdmpc2_cfg(obs_dim=obs_dim, action_dim=act_dim, device="cpu", horizon=3)
    return WorldModel(cfg).to("cpu"), cfg


def test_target_starts_equal_then_diverges_on_soft_update():
    wm, cfg = _wm()
    # perturb online Qs
    for p in wm._Qs.parameters():
        p.data.add_(torch.randn_like(p))
    before = [tp.clone() for tp in wm._target_Qs.parameters()]
    wm.soft_update_target_Q()
    after = list(wm._target_Qs.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))  # target moved


def test_detach_path_no_qweight_grad():
    wm, cfg = _wm()
    z = torch.randn(8, cfg.latent_dim, requires_grad=True)
    a = torch.randn(8, cfg.action_dim)
    q = wm.Q(z, a, task=None, return_type="avg", detach=True)
    q.sum().backward()
    assert z.grad is not None                       # grad reaches input
    assert all(p.grad is None for p in wm._Qs.parameters())  # not the weights
