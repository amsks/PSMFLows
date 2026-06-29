import torch, types
from agents.tdmpc2.agent import TDMPC2Agent
from evals.phase_probe import tdmpc2_latent_phase_probe


def test_probe_returns_expected_keys():
    a = TDMPC2Agent(types.SimpleNamespace(shape=(19,)), 5, device="cpu", horizon=3)
    B, H, P = 32, 3, 40
    batch = {
        "observation": torch.randn(B, H, 19),
        "action": torch.randn(B, H, 5).clamp(-1, 1),
        "next": {"observation": torch.randn(B, H, 19),
                 "physics": torch.randn(B, H, P)},
    }
    goal = torch.zeros(3)
    out = tdmpc2_latent_phase_probe(a, batch, goal, rollout_len=2)
    assert "cube_xyz_mse" in out
    assert "open_loop_cube_mse_by_step" in out
    assert len(out["open_loop_cube_mse_by_step"]) == 2
