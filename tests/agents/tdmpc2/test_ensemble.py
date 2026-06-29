import torch
import torch.nn as nn
from agents.tdmpc2.layers import Ensemble


def _make_members(n, in_dim, out_dim):
    torch.manual_seed(0)
    return [nn.Sequential(nn.Linear(in_dim, 8), nn.ReLU(), nn.Linear(8, out_dim)) for _ in range(n)]


def test_vmap_ensemble_matches_naive_loop():
    n, in_dim, out_dim, B = 5, 6, 3, 4
    members = _make_members(n, in_dim, out_dim)
    ens = Ensemble(members)
    x = torch.randn(B, in_dim)
    out = ens(x)                                  # [n, B, out_dim]
    assert out.shape == (n, B, out_dim)
    ref = torch.stack([m(x) for m in members], 0)  # naive loop
    assert torch.allclose(out, ref, atol=1e-6), (out - ref).abs().max()


def test_ensemble_len():
    members = _make_members(3, 4, 2)
    assert len(Ensemble(members)) == 3


def test_ensemble_gradients_reach_parameters():
    # Regression: the optimizer trains Ensemble.parameters(); backward MUST
    # populate grads on them (a forward-only vmap over disconnected stacked
    # params would leave them frozen -> value_loss never moves).
    members = _make_members(3, 4, 2)
    ens = Ensemble(members)
    params = list(ens.parameters())
    assert len(params) > 0
    out = ens(torch.randn(5, 4))
    out.sum().backward()
    assert all(p.grad is not None for p in params)
    assert any(p.grad.abs().sum() > 0 for p in params)
