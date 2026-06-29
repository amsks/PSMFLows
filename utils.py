# Adapted from Meta Platforms, Inc. metamotivo/nn_models.py
# CC BY-NC 4.0 license
#
# Changes from original:
#   - Removed DrQEncoder, Augmentator (pixel modules — add back when needed)
#   - Removed VForwardMap, VectorField (not used in FB baseline)
#   - Kept all state-based modules verbatim: BackwardMap, ForwardMap, Actor,
#     NoiseConditionedActor, DenseParallel, ParallelLayerNorm, TruncatedNormal, Norm
#   - Kept all config classes for the modules above

import math
import numbers
import typing as tp

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn
from torch.distributions.utils import _standard_normal

from base_config import BaseConfig

##########################
# Initialization utils
##########################


def parallel_orthogonal_(tensor, gain=1):
    if tensor.ndimension() == 2:
        tensor = nn.init.orthogonal_(tensor, gain=gain)
        return tensor
    if tensor.ndimension() < 3:
        raise ValueError("Only tensors with 3 or more dimensions are supported")
    n_parallel = tensor.size(0)
    rows = tensor.size(1)
    cols = tensor.numel() // n_parallel // rows
    flattened = tensor.new(n_parallel, rows, cols).normal_(0, 1)

    qs = []
    for flat_tensor in torch.unbind(flattened, dim=0):
        if rows < cols:
            flat_tensor.t_()
        q, r = torch.linalg.qr(flat_tensor)
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph
        if rows < cols:
            q.t_()
        qs.append(q)

    qs = torch.stack(qs, dim=0)
    with torch.no_grad():
        tensor.view_as(qs).copy_(qs)
        tensor.mul_(gain)
    return tensor


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, DenseParallel):
        gain = nn.init.calculate_gain("relu")
        parallel_orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif hasattr(m, "reset_parameters"):
        m.reset_parameters()


##########################
# Update utils
##########################


def _soft_update_params(net_params: tp.Any, target_net_params: tp.Any, tau: float):
    torch._foreach_mul_(target_net_params, 1 - tau)
    torch._foreach_add_(target_net_params, net_params, alpha=tau)


class eval_mode:
    def __init__(self, *models) -> None:
        self.models = models
        self.prev_states: list = []

    def __enter__(self) -> None:
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args) -> None:
        for model, state in zip(self.models, self.prev_states):
            model.train(state)


##########################
# Creation utils
##########################


class ForwardArchiConfig(BaseConfig):
    name: tp.Literal["ForwardArchi"] = "ForwardArchi"
    hidden_dim: int = 1024
    model: tp.Literal["simple"] = "simple"
    hidden_layers: int = 1
    embedding_layers: int = 2
    num_parallel: int = 2
    ensemble_mode: tp.Literal["batch"] = "batch"

    def build(self, obs_space, z_dim: int, action_dim, output_dim=None) -> torch.nn.Module:
        if self.ensemble_mode == "batch":
            return _build_batch_forward(self, obs_space, z_dim, action_dim, output_dim)
        raise ValueError(f"Unsupported ensemble_mode {self.ensemble_mode}")


def _build_batch_forward(cfg, obs_space, z_dim, action_dim, output_dim=None):
    if cfg.model == "simple":
        return ForwardMap(obs_space, z_dim, action_dim, cfg, output_dim=output_dim)
    raise ValueError(f"Unsupported forward_map model {cfg.model}")


class ActorArchiConfig(BaseConfig):
    name: tp.Literal["actor"] = "actor"
    model: tp.Literal["simple"] = "simple"
    hidden_dim: int = 1024
    hidden_layers: int = 1
    embedding_layers: int = 2

    def build(self, obs_space, z_dim, action_dim):
        if self.model == "simple":
            return Actor(obs_space, z_dim, action_dim, self)
        raise ValueError(f"Unsupported actor model {self.model}")


def linear(input_dim, output_dim, num_parallel=1):
    if num_parallel > 1:
        return DenseParallel(input_dim, output_dim, n_parallel=num_parallel)
    return nn.Linear(input_dim, output_dim)


def layernorm(input_dim, num_parallel=1):
    if num_parallel > 1:
        return ParallelLayerNorm([input_dim], n_parallel=num_parallel)
    return nn.LayerNorm(input_dim)


##########################
# Simple MLP models
##########################


class BackwardArchiConfig(BaseConfig):
    name: tp.Literal["BackwardArchi"] = "BackwardArchi"
    hidden_dim: int = 256
    hidden_layers: int = 2
    norm: bool = True

    def build(self, obs_space, z_dim: int):
        return BackwardMap(obs_space, z_dim, self)


class BackwardMap(nn.Module):
    def __init__(self, obs_space, z_dim, cfg: BackwardArchiConfig) -> None:
        super().__init__()
        self.cfg: BackwardArchiConfig = cfg
        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        seq = [nn.Linear(obs_space.shape[0], cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.Tanh()]
        for _ in range(cfg.hidden_layers - 1):
            seq += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [nn.Linear(cfg.hidden_dim, z_dim)]
        if cfg.hidden_layers == 0:
            seq = [nn.Linear(obs_space.shape[0], z_dim)]
        if cfg.norm:
            seq += [Norm()]
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def simple_embedding(input_dim, hidden_dim, hidden_layers, num_parallel=1):
    assert hidden_layers >= 2, "must have at least 2 embedding layers"
    seq = [linear(input_dim, hidden_dim, num_parallel), layernorm(hidden_dim, num_parallel), nn.Tanh()]
    for _ in range(hidden_layers - 2):
        seq += [linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]
    seq += [linear(hidden_dim, hidden_dim // 2, num_parallel), nn.ReLU()]
    return nn.Sequential(*seq)


class ForwardMap(nn.Module):
    def __init__(self, obs_space, z_dim, action_dim, cfg: ForwardArchiConfig, output_dim=None) -> None:
        super().__init__()
        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = obs_space.shape[0]
        self.cfg = cfg
        self.z_dim = z_dim
        self.num_parallel = cfg.num_parallel
        self.hidden_dim = cfg.hidden_dim

        self.embed_z = simple_embedding(obs_dim + z_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)
        self.embed_sa = simple_embedding(obs_dim + action_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)

        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim, cfg.num_parallel), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, output_dim if output_dim else z_dim, cfg.num_parallel)]
        self.Fs = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        # Returns [num_parallel, B, z_dim] when num_parallel > 1, else [B, z_dim]
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            action = action.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))
        sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1))
        return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))


class SimpleActorArchiConfig(ActorArchiConfig):
    name: tp.Literal["simple"] = "simple"
    model: tp.Literal["simple"] = "simple"

    def build(self, obs_space, z_dim: int, action_dim: int) -> "Actor":
        return Actor(obs_space, z_dim, action_dim, self)


class Actor(nn.Module):
    def __init__(self, obs_space, z_dim, action_dim, cfg: SimpleActorArchiConfig) -> None:
        super().__init__()
        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = obs_space.shape[0]
        self.cfg: SimpleActorArchiConfig = cfg

        self.embed_z = simple_embedding(obs_dim + z_dim, cfg.hidden_dim, cfg.embedding_layers)
        self.embed_s = simple_embedding(obs_dim, cfg.hidden_dim, cfg.embedding_layers)

        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, action_dim)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, std: float) -> "TruncatedNormal":
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))
        s_embedding = self.embed_s(obs)
        mu = torch.tanh(self.policy(torch.cat([s_embedding, z_embedding], dim=-1)))
        return TruncatedNormal(mu, torch.ones_like(mu) * std)


class NoiseConditionedActorArchiConfig(BaseConfig):
    name: tp.Literal["noise_conditioned_actor"] = "noise_conditioned_actor"
    model: tp.Literal["simple"] = "simple"
    hidden_dim: int = 1024
    hidden_layers: int = 1
    embedding_layers: int = 2

    def build(self, obs_space, z_dim: int, action_dim: int) -> "NoiseConditionedActor":
        return NoiseConditionedActor(obs_space, z_dim, action_dim, self)


class NoiseConditionedActor(nn.Module):
    def __init__(self, obs_space, z_dim, action_dim, cfg: NoiseConditionedActorArchiConfig) -> None:
        super().__init__()
        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = obs_space.shape[0]
        self.cfg: NoiseConditionedActorArchiConfig = cfg
        self.embed_z = simple_embedding(obs_dim + z_dim + action_dim, cfg.hidden_dim, cfg.embedding_layers)
        self.embed_s = simple_embedding(obs_dim + action_dim, cfg.hidden_dim, cfg.embedding_layers)

        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, action_dim)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        z_embedding = self.embed_z(torch.cat([obs, z, noise], dim=-1))
        s_embedding = self.embed_s(torch.cat([obs, noise], dim=-1))
        return torch.tanh(self.policy(torch.cat([s_embedding, z_embedding], dim=-1)))


##########################
# Helper modules
##########################


class DenseParallel(nn.Module):
    def __init__(self, in_features, out_features, n_parallel, bias=True, device=None, dtype=None, reset_params=True):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_parallel = n_parallel
        if n_parallel is None or n_parallel == 1:
            self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs)) if bias else None
        else:
            self.weight = nn.Parameter(torch.empty((n_parallel, in_features, out_features), **factory_kwargs))
            self.bias = nn.Parameter(torch.empty((n_parallel, 1, out_features), **factory_kwargs)) if bias else None
            if self.bias is None:
                raise NotImplementedError
        if reset_params:
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.n_parallel is None or self.n_parallel == 1:
            return F.linear(input, self.weight, self.bias)
        return torch.baddbmm(self.bias, input, self.weight)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, n_parallel={self.n_parallel}"


class ParallelLayerNorm(nn.Module):
    def __init__(self, normalized_shape, n_parallel, eps=1e-5, elementwise_affine=True, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = [normalized_shape]
        assert len(normalized_shape) == 1
        self.n_parallel = n_parallel
        self.normalized_shape = list(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            if n_parallel is None or n_parallel == 1:
                self.weight = nn.Parameter(torch.empty([*self.normalized_shape], **factory_kwargs))
                self.bias = nn.Parameter(torch.empty([*self.normalized_shape], **factory_kwargs))
            else:
                self.weight = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
                self.bias = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        norm_input = F.layer_norm(input, self.normalized_shape, None, None, self.eps)
        if self.elementwise_affine:
            return (norm_input * self.weight) + self.bias
        return norm_input


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        return x - x.detach() + clamped_x.detach()

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        return self._clamp(self.loc + eps)


class Norm(nn.Module):
    """Normalise to ||x|| = sqrt(d), matching the metamotivo z-normalisation convention."""
    def forward(self, x) -> torch.Tensor:
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)


##########################
# Vector field (flow matching)
##########################


class SimpleVectorFieldArchiConfig(BaseConfig):
    name: tp.Literal["SimpleVectorFieldArchi"] = "SimpleVectorFieldArchi"
    model: tp.Literal["simple"] = "simple"
    hidden_dim: int = 1024
    hidden_layers: int = 1

    def build(self, obs_space, action_dim: int) -> "VectorField":
        return VectorField(obs_space, action_dim, self)


class VectorField(nn.Module):
    """Unconditional vector field for flow matching (no goal z).
    obs + action + t → velocity."""

    def __init__(self, obs_space, action_dim, cfg: SimpleVectorFieldArchiConfig) -> None:
        super().__init__()
        self.cfg: SimpleVectorFieldArchiConfig = cfg
        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = obs_space.shape[0]
        # +1 for the time dimension t
        seq = [linear(obs_dim + action_dim + 1, cfg.hidden_dim), nn.GELU()]
        for _ in range(cfg.hidden_layers - 1):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU()]
        seq += [linear(cfg.hidden_dim, action_dim)]
        self.net = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action, t], dim=-1))