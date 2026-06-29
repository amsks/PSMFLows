import numpy as np
import torch
from agents.psm.proto_sampler import ProtoBehaviorSampler


def _reference_table(action_dim, max_log_seed):
    # Transcribed from PSM SamplingSeedActor.__init__ (psm.py:35-43)
    max_seed = 2 ** max_log_seed + 20000
    seed_to_action = []
    for i in range(max_seed):
        torch.random.manual_seed(i)
        action = (torch.rand(size=(action_dim,)).unsqueeze(0) - 1) * 2
        seed_to_action.append(action)
    return np.array(seed_to_action).squeeze(), max_seed


def test_table_matches_reference():
    s = ProtoBehaviorSampler(action_dim=5, max_log_seed=8, batch_size=4)  # small for speed
    ref_table, ref_max = _reference_table(5, 8)
    assert s.max_seed == ref_max
    assert np.allclose(s.seed_to_action, ref_table)


def test_forward_indexing_matches_reference():
    s = ProtoBehaviorSampler(action_dim=5, max_log_seed=8, batch_size=4)
    z = torch.randint(0, 2, (4, 8)).float()
    obs_hash = torch.tensor([0, 1, 2, 3])
    powers = torch.tensor([2 ** i for i in range(8)][::-1]).repeat(4, 1)
    seed_long = (z * powers).sum(1)
    final_seed = ((seed_long + obs_hash.reshape(-1)) % s.max_seed).cpu().numpy().astype(np.int32)
    expected = torch.FloatTensor(s.seed_to_action[final_seed])
    out = s(obs_hash, z)
    assert torch.equal(out, expected)
