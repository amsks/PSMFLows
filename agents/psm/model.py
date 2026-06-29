"""agents/psm/model.py — PSMModel assembling the PSM networks + targets.

State-only for now (pixel encoders wired later). The encoder/normalizer block is
copied from agents/fb/model.py.FBModel.__init__ so the construction (and the
Identity defaults for state) matches FB exactly.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from agents.psm.psm_nets import PhiMap, PsiMap, Actor, weight_init
from agents.psm.proto_sampler import ProtoBehaviorSampler
from nn_models import IdentityNNConfig, eval_mode
from normalizers import IdentityNormalizerConfig


class PSMModel(nn.Module):
    def __init__(self, obs_space, action_dim, z_dim=128, max_log_seed=16, batch_size=1024,
                 norm_z=True, phi_input="s", phi_cfg=None, sf_cfg=None, actor_cfg=None,
                 actor_kind="td3", obs_normalizer_cfg=None, rgb_encoder_cfg=None,
                 augmentator_cfg=None, num_parallel=2, device="cpu", amp=False):
        super().__init__()
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.max_log_seed = max_log_seed
        self.batch_size = batch_size
        self.norm_z = norm_z
        self.phi_input = phi_input
        self.num_parallel = num_parallel
        self.device = device
        self.device_type = str(device).split(":")[0]
        self.amp = amp
        self.amp_dtype = torch.bfloat16

        # === Encoder/normalizer construction copied from FBModel.__init__ ===
        # Identity defaults for state (matches FB). For state these are Identity
        # and obs_dim is derived from the fw-encoder output_space.
        if rgb_encoder_cfg is None:
            rgb_encoder_cfg = IdentityNNConfig()
        if augmentator_cfg is None:
            augmentator_cfg = IdentityNNConfig()
        if obs_normalizer_cfg is None:
            obs_normalizer_cfg = IdentityNormalizerConfig()

        self._obs_normalizer = obs_normalizer_cfg.build(obs_space)
        self._bw_encoder = rgb_encoder_cfg.build(obs_space)
        self._augmentator = augmentator_cfg.build(obs_space)
        self._fw_encoder = rgb_encoder_cfg.build(obs_space)

        obs_dim = self._fw_encoder.output_space.shape[0]
        # === end copied block ===

        goal_dim = {"s": obs_dim, "as": obs_dim + action_dim,
                    "sas": 2 * obs_dim + action_dim, "ss": 2 * obs_dim}[phi_input]
        self.goal_dim = goal_dim
        phi_cfg = phi_cfg or {}; sf_cfg = sf_cfg or {}; actor_cfg = actor_cfg or {}

        self.phi = PhiMap(goal_dim, z_dim, phi_cfg.get("hidden_dim", 256),
                          phi_cfg.get("hidden_layers", 2), phi_cfg.get("norm", True),
                          phi_cfg.get("batch_norm", False))
        self.sf_psi = PsiMap(obs_dim, z_dim, action_dim, sf_cfg.get("hidden_dim", 1024),
                             sf_cfg.get("hidden_layers", 1), sf_cfg.get("embedding_layers", 2), num_parallel)
        self.psm_psi = PsiMap(obs_dim, max_log_seed, action_dim, sf_cfg.get("hidden_dim", 1024),
                              sf_cfg.get("hidden_layers", 1), sf_cfg.get("embedding_layers", 2),
                              num_parallel, output_dim=z_dim)
        self.actor = (Actor(obs_dim, z_dim, action_dim, actor_cfg.get("hidden_dim", 1024),
                            actor_cfg.get("hidden_layers", 1), actor_cfg.get("embedding_layers", 2))
                      if actor_kind == "td3" else None)
        self.proto_sampler = ProtoBehaviorSampler(action_dim, max_log_seed, batch_size, device)

        nets = {"sf_psi": self.sf_psi, "psm_psi": self.psm_psi, "phi": self.phi}
        if self.actor is not None: nets["actor"] = self.actor
        for net in nets.values(): net.apply(weight_init)
        self.target_phi = copy.deepcopy(self.phi)
        self.target_sf_psi = copy.deepcopy(self.sf_psi)
        self.target_psm_psi = copy.deepcopy(self.psm_psi)
        self.num_parallel_scaling = num_parallel ** 2 - num_parallel
        self.actor_std = actor_cfg.get("std", 0.2)
        self.stddev_clip = actor_cfg.get("stddev_clip", 0.3)
        self.to(device)

    def _normalize(self, obs):
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return self._obs_normalizer(obs)

    def project_z(self, z):
        if self.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def int_to_binary_array(self, int_vector, num_bits):
        return ((int_vector[:, None] & (1 << np.arange(num_bits))) > 0).astype(int)

    def sample_z(self, size, device="cpu"):
        z = torch.randn((size, self.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def sample_z_psm(self, size, device="cpu"):
        z_np = np.random.randint(0, 2 ** self.max_log_seed, (size,))
        binary = self.int_to_binary_array(z_np, self.max_log_seed)
        return torch.FloatTensor(binary).to(device)

    def get_targets_uncertainty(self, preds, dim=0):
        preds_mean = preds.mean(dim=dim)
        d1 = preds.unsqueeze(dim=dim); d2 = preds.unsqueeze(dim=dim + 1)
        preds_unc = torch.abs(d1 - d2).sum(dim=(dim, dim + 1)) / self.num_parallel_scaling
        return preds_mean, preds_unc

    @torch.no_grad()
    def reward_inference(self, next_obs, reward, weight=None):
        assert self.phi_input == "s", "reward_inference supports phi_input='s'"
        next_obs = next_obs.to(self.device).float(); reward = reward.to(self.device).float()
        with eval_mode(self):
            goal = self._bw_encoder(self._normalize(next_obs))  # state: Identity(normalized obs)
            phi = self.phi(goal)
        z = torch.matmul(reward.T, phi) / phi.shape[0]
        return self.project_z(z)

    @torch.no_grad()
    def act(self, obs, z, mean=True, goal=None):
        # state: encoder Identity. Mirror FBModel.act. TD3 actor returns a TruncatedNormal.
        with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp):
            obs = self._fw_encoder(self._normalize(obs))
            dist = self.actor(obs, z, self.actor_std)
        return dist.mean.float() if mean else dist.sample(clip=self.stddev_clip).float()
