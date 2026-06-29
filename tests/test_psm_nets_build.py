import math
import torch
from agents.psm.psm_nets import PhiMap, PsiMap, SimpleActor, Actor, mlp, parallel_mlp


def test_phimap_shape_and_norm():
    phi = PhiMap(goal_dim=40, z_dim=128, hidden_dim=256, hidden_layers=2, norm=True)
    out = phi(torch.randn(8, 40))
    assert out.shape == (8, 128)
    assert torch.allclose(out.norm(dim=-1), torch.full((8,), math.sqrt(128)), atol=1e-4)


def test_psimap_parallel_output_dim():
    sf = PsiMap(obs_dim=40, z_dim=128, action_dim=5, hidden_dim=1024,
                hidden_layers=1, embedding_layers=2, num_parallel=2)
    out = sf(torch.randn(8, 40), torch.randn(8, 128), torch.randn(8, 5))
    assert out.shape == (2, 8, 128)
    psm = PsiMap(obs_dim=40, z_dim=16, action_dim=5, hidden_dim=1024,
                 hidden_layers=1, embedding_layers=2, num_parallel=2, output_dim=128)
    out2 = psm(torch.randn(8, 40), torch.randn(8, 16), torch.randn(8, 5))
    assert out2.shape == (2, 8, 128)


def test_actor_returns_truncated_normal():
    act = Actor(obs_dim=40, z_dim=128, action_dim=5, hidden_dim=1024,
                hidden_layers=1, embedding_layers=2)
    dist = act(torch.randn(8, 40), torch.randn(8, 128), 0.2)
    a = dist.sample(clip=0.3)
    assert a.shape == (8, 5) and a.abs().max() <= 1.0
