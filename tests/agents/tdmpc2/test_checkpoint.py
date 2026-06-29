import torch, types
from agents.tdmpc2.agent import TDMPC2Agent


def _agent():
    return TDMPC2Agent(types.SimpleNamespace(shape=(19,)), 5, device="cpu", horizon=3, batch_size=8)


def test_state_dict_save_load_roundtrip(tmp_path):
    a = _agent()
    # perturb so saved weights are non-default
    for p in a.core.model._dynamics.parameters():
        p.data.add_(torch.randn_like(p))
    a.core.scale.value.add_(0.3)
    path = tmp_path / "ckpt.pt"
    a.save(str(path))

    b = _agent()
    b.load(str(path))
    for pa, pb in zip(a.core.model.parameters(), b.core.model.parameters()):
        assert torch.equal(pa, pb)
    assert torch.equal(a.core.scale.value, b.core.scale.value)
