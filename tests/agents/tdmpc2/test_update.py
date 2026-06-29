import torch, types
from agents.tdmpc2.agent import TDMPC2Agent


def _agent(obs_dim=19, act_dim=5, H=3, B=16):
    return TDMPC2Agent(types.SimpleNamespace(shape=(obs_dim,)), act_dim,
                       device="cpu", horizon=H, batch_size=B)


def _windowed_batch(B=16, H=3, obs_dim=19, act_dim=5, P=40):
    rand = torch.randn
    return {
        "observation": rand(B, H, obs_dim),
        "action": rand(B, H, act_dim).clamp(-1, 1),
        "physics": rand(B, H, P),
        "next": {
            "observation": rand(B, H, obs_dim),
            "physics": rand(B, H, P),
            "terminated": torch.zeros(B, H, 1),
        },
    }


def test_update_runs_and_returns_float_metrics():
    a = _agent()
    metrics = a.update(_windowed_batch(), step=0)
    assert "consistency_loss" in metrics and "value_loss" in metrics
    assert all(isinstance(v, float) for v in metrics.values())


def test_update_changes_params():
    a = _agent()
    p0 = [p.clone() for p in a.core.model._dynamics.parameters()]
    for s in range(3):
        a.update(_windowed_batch(), step=s)
    p1 = list(a.core.model._dynamics.parameters())
    assert any(not torch.equal(b, x) for b, x in zip(p0, p1))
