"""Coverage-balanced FB: goal-agnostic rho' reweight.
See docs/superpowers/specs/2026-05-25-coverage-balanced-fb.md.

Two intended variants (per the review):
  - PURE rho'-FB (weight_diag=False, default): off-diagonal (g~rho'), orthonormality
    and reward-cov are weighted by w; the diagonal (Bellman/Dirac -E[F^T B(s')]) is
    left UNWEIGHTED (its density ratio cancels under L2(rho')).
  - coverage-PRIORITIZED FB (weight_diag=True): additionally weights the diagonal by
    w_i = extra TD pressure on rare achieved next-states (a hybrid, not a pure measure
    change). w=None reproduces stock FB exactly.
"""
import inspect

import numpy as np
import torch

from agents.fb.agent import coverage_weights, fb_successor_terms, ortho_cov


def _density():  # 2x2x2 cube-xyz density: cell (0,0,0) common, (1,1,1) rare
    H = torch.zeros(2, 2, 2)
    H[0, 0, 0] = 0.9
    H[1, 1, 1] = 0.001
    H = H / H.sum()
    edges = [torch.tensor([0.0, 0.5, 1.0]) for _ in range(3)]
    return (H, edges)


# --- byte-identical: w=None == w=ones for every term --------------------------------

def test_successor_ones_equal_none():
    torch.manual_seed(0)
    diff = torch.randn(3, 5, 5); od = 1.0 - torch.eye(5); s = od.sum()
    o0, d0 = fb_successor_terms(diff, od, s, None)
    o1, d1 = fb_successor_terms(diff, od, s, torch.ones(5))
    assert torch.allclose(o0, o1) and torch.allclose(d0, d1)


def test_ortho_cov_ones_equal_none():
    torch.manual_seed(0); B = torch.randn(6, 4)
    assert torch.allclose(ortho_cov(B, None), ortho_cov(B, torch.ones(6)))


# --- pure rho' (default): off-diagonal weighted, diagonal UNweighted ----------------

def test_pure_rho_weights_offdiag_not_diag():
    torch.manual_seed(1)
    diff = torch.randn(2, 4, 4); od = 1.0 - torch.eye(4); s = od.sum()
    w = torch.tensor([3.0, 0.4, 0.3, 0.3]); w = w / w.mean()
    o_w, d_w = fb_successor_terms(diff, od, s, w)          # weight_diag=False (default)
    o_n, d_n = fb_successor_terms(diff, od, s, None)
    assert not torch.allclose(o_w, o_n)                    # off-diagonal IS reweighted
    assert torch.allclose(d_w, d_n)                        # diagonal is NOT (pure rho')


def test_prioritized_weights_diag():
    torch.manual_seed(2)
    diff = torch.randn(2, 4, 4); od = 1.0 - torch.eye(4); s = od.sum()
    w = torch.tensor([3.0, 0.4, 0.3, 0.3]); w = w / w.mean()
    _, d_w = fb_successor_terms(diff, od, s, w, weight_diag=True)
    _, d_n = fb_successor_terms(diff, od, s, None)
    assert not torch.allclose(d_w, d_n)                    # prioritized: diagonal reweighted


# --- coverage_weights ---------------------------------------------------------------

def test_coverage_weights_rare_gets_more_weight():
    cube = torch.tensor([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    w = coverage_weights(cube, _density(), alpha=1.0, clip=10.0)
    assert torch.isfinite(w).all() and w[1] > w[0] and abs(w.mean().item() - 1.0) < 1e-5


def test_coverage_weights_alpha_zero_uniform():
    cube = torch.tensor([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    assert torch.allclose(coverage_weights(cube, _density(), 0.0, 10.0), torch.ones(2))


def test_coverage_weights_two_sided_clip():
    """raw w clamped to [1/clip, clip] => post-norm max/min ratio == clip^2 when both
    clips bind (dense cell floored at 1/clip, rare capped at clip)."""
    H = torch.zeros(2, 2, 2); H[0, 0, 0] = 0.7; H[0, 0, 1] = 0.2; H[1, 1, 1] = 0.1; H /= H.sum()
    edges = [torch.tensor([0.0, 0.5, 1.0]) for _ in range(3)]
    cube = torch.tensor([[0.25, 0.25, 0.25],   # dense  -> raw 0.286, floored to 1/clip
                         [0.25, 0.25, 0.75],   # medium
                         [0.25, 0.25, 0.75],
                         [0.75, 0.75, 0.75]])  # rare   -> capped at clip
    clip = 2.0
    w = coverage_weights(cube, (H, edges), alpha=1.0, clip=clip)
    assert (w.max() / w.min()).item() <= clip * clip + 1e-3
    assert w.min().item() > 0.0


def test_coverage_weights_goal_agnostic():
    params = set(inspect.signature(coverage_weights).parameters)
    assert not (params & {"goal", "goals", "task", "z", "reward"})


# --- orthonormality reweight identity (the centerpiece) -----------------------------

def test_ortho_cov_weighted_second_moment():
    torch.manual_seed(1); B = torch.randn(6, 4); w = torch.rand(6) + 0.5; w = w / w.mean()
    Cov = ortho_cov(B, w)
    assert torch.allclose(Cov.diag().mean(), (w * (B * B).sum(-1)).mean(), atol=1e-6)
    assert not torch.allclose(Cov, ortho_cov(B, None))


# --- agent wiring: defaults, fail-loud, weights-from-physics ------------------------

def _tiny(**extra):
    import gymnasium
    from agents.fb.agent import FBAgent
    from nn_models import IdentityNNConfig
    from normalizers import IdentityNormalizerConfig
    return FBAgent(
        obs_space=gymnasium.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
        action_dim=2, z_dim=8, L_dim=8,
        obs_normalizer_cfg=IdentityNormalizerConfig(),
        rgb_encoder_cfg=IdentityNNConfig(),
        augmentator_cfg=IdentityNNConfig(),
        device="cpu", **extra,
    )


def test_agent_defaults_off():
    a = _tiny()
    assert a.reweight_alpha == 0.0 and a._rho is None
    assert a.weight_diag is False and a.weight_z is False
    assert a._coverage_weights({"next": {"physics": torch.rand(4, 20)}}) is None


def test_agent_fail_loud_without_density_path():
    import pytest
    with pytest.raises(ValueError):
        _tiny(reweight_alpha=1.0, reweight_density_path=None)


def test_agent_coverage_weights_from_physics(tmp_path):
    p = tmp_path / "cube_density.npz"
    H, edges = _density()
    np.savez(p, H=H.numpy(), ex=edges[0].numpy(), ey=edges[1].numpy(), ez=edges[2].numpy())
    a = _tiny(reweight_alpha=1.0, reweight_density_path=str(p), weight_diag=True, weight_z=True)
    assert a._rho is not None and a.weight_diag is True and a.weight_z is True
    w = a._coverage_weights({"next": {"physics": torch.rand(8, 20)}})
    assert w is not None and w.shape == (8,) and torch.isfinite(w).all()
    assert abs(w.mean().item() - 1.0) < 1e-5
