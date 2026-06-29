"""agents/psm/proto_sampler.py — port of PSM's SamplingSeedActor (psm.py:29-53).

A deterministic pseudo-random behavior policy keyed by (row_index, binary z) used as
the target next-action when learning the proto successor net psm_psi. The reference
hardcodes 'cuda'; we take a device. Table/indexing reproduced byte-for-byte.
"""
import numpy as np
import torch
import torch.nn as nn


class ProtoBehaviorSampler(nn.Module):
    def __init__(self, action_dim: int, max_log_seed: int, batch_size: int, device: str = "cpu") -> None:
        super().__init__()
        self.action_dim = action_dim
        self.device = device
        # Register as a non-persistent buffer so module.to(device) moves it, while
        # keeping it OUT of state_dict (it's deterministically rebuilt in __init__;
        # the reference does not checkpoint it).
        self.register_buffer(
            "powers",
            torch.tensor([2 ** i for i in range(max_log_seed)][::-1]).to(device).repeat(batch_size, 1),
            persistent=False,
        )
        self.max_seed = 2 ** max_log_seed + 20000
        seed_to_action = []
        for i in range(self.max_seed):
            torch.random.manual_seed(i)
            action = (torch.rand(size=(self.action_dim,)).unsqueeze(0) - 1) * 2
            seed_to_action.append(action)
        self.seed_to_action = np.array(seed_to_action).squeeze()

    def forward(self, obs_hash: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        assert z.shape[0] == self.powers.shape[0], "z batch must match the configured batch_size"
        seed_long = (z * self.powers).sum(1)
        final_seed = (seed_long + obs_hash.reshape(-1)) % self.max_seed
        actions = self.seed_to_action[final_seed.cpu().numpy().astype(np.int32)]
        return torch.FloatTensor(actions).to(self.device)
