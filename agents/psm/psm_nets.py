"""agents/psm/psm_nets.py — VERBATIM port of the PSM reference networks.

Sources (copied unchanged so the bit-exact reference-equivalence test passes):
  models/modules.py         -> _L2, _nl, Norm, mlp
  models/parallel_modules.py-> weight_init, parallel_orthogonal_, DenseParallel,
                               ParallelLayerNorm, _parallel_nl, parallel_mlp
  models/psm_models.py      -> PhiMap, build_embedding(_relu), PsiMap, SimpleActor, Actor
Only change vs reference: TruncatedNormal import path. DO NOT modify the math/op-order.
"""
import math
import numbers
import typing as tp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from agents.psm._truncated_normal import TruncatedNormal


class _L2(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        y = math.sqrt(self.dim) * F.normalize(x, dim=1)
        return y


def _nl(name: str, dim: int) -> tp.List[nn.Module]:
    """Returns a non-linearity given name and dimension"""
    if name == "irelu":
        return [nn.ReLU(inplace=True)]
    if name == "relu":
        return [nn.ReLU()]
    if name == "ntanh":
        return [nn.LayerNorm(dim), nn.Tanh()]
    if name == "layernorm":
        return [nn.LayerNorm(dim)]
    if name == "tanh":
        return [nn.Tanh()]
    if name == "bnorm":
        return [nn.BatchNorm1d(dim, affine=False)]
    if name == "norm":
        return [Norm()]
    if name == "L2":
        return [_L2(dim)]
    raise ValueError(f"Unknown non-linearity {name}")


class Norm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)


def mlp(*layers: tp.Sequence[tp.Union[int, str]]) -> nn.Sequential:
    assert len(layers) >= 2
    sequence: tp.List[nn.Module] = []
    assert isinstance(layers[0], int), "First input must provide the dimension"
    prev_dim: int = layers[0]
    for layer in layers[1:]:
        if isinstance(layer, str):
            sequence.extend(_nl(layer, prev_dim))
        else:
            assert isinstance(layer, int)
            sequence.append(nn.Linear(prev_dim, layer))
            prev_dim = layer
    return nn.Sequential(*sequence)


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
    elif hasattr(m, "reset_parameters"):
        m.reset_parameters()


# Initialization for parallel layers
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

        # Compute the qr factorization
        q, r = torch.linalg.qr(flat_tensor)
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
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


class DenseParallel(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_parallel: int,
        bias: bool = True,
        device=None,
        dtype=None,
        reset_params=True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(DenseParallel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_parallel = n_parallel
        if n_parallel is None or (n_parallel == 1):
            self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.weight = nn.Parameter(
                torch.empty((n_parallel, in_features, out_features), **factory_kwargs)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.empty((n_parallel, 1, out_features), **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
            if self.bias is None:
                raise NotImplementedError
        if reset_params:
            self.reset_parameters()

    def load_module_list_weights(self, module_list) -> None:
        with torch.no_grad():
            assert len(module_list) == self.n_parallel
            weight_list = [m.weight.T for m in module_list]
            target_weight = torch.stack(weight_list, dim=0)
            self.weight.data.copy_(target_weight.data)
            if self.bias:
                bias_list = [ln.bias.unsqueeze(0) for ln in module_list]
                target_bias = torch.stack(bias_list, dim=0)
                self.bias.data.copy_(target_bias.data)

    # TODO why do these layers have their own reset scheme?
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.n_parallel is None or (self.n_parallel == 1):
            return F.linear(input, self.weight, self.bias)
        else:
            return torch.baddbmm(self.bias, input, self.weight)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, n_parallel={}, bias={}".format(
            self.in_features, self.out_features, self.n_parallel, self.bias is not None
        )


class ParallelLayerNorm(nn.Module):
    def __init__(self, normalized_shape, n_parallel, eps=1e-5, elementwise_affine=True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ParallelLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = [normalized_shape, ]
        assert len(normalized_shape) == 1
        self.n_parallel = n_parallel
        self.normalized_shape = list(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            if n_parallel is None or (n_parallel == 1):
                self.weight = nn.Parameter(torch.empty([*self.normalized_shape], **factory_kwargs))
                self.bias = nn.Parameter(torch.empty([*self.normalized_shape], **factory_kwargs))
            else:
                self.weight = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
                self.bias = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def load_module_list_weights(self, module_list) -> None:
        with torch.no_grad():
            assert len(module_list) == self.n_parallel
            if self.elementwise_affine:
                ln_weights = [ln.weight.unsqueeze(0) for ln in module_list]
                ln_biases = [ln.bias.unsqueeze(0) for ln in module_list]
                target_ln_weights = torch.stack(ln_weights, dim=0)
                target_ln_bias = torch.stack(ln_biases, dim=0)
                self.weight.data.copy_(target_ln_weights.data)
                self.bias.data.copy_(target_ln_bias.data)


    def forward(self, input):
        norm_input = F.layer_norm(
            input, self.normalized_shape, None, None, self.eps)
        if self.elementwise_affine:
            return (norm_input * self.weight) + self.bias
        else:
            return norm_input

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


def _parallel_nl(name: str, dim: int, n_parallel: int) -> tp.List[nn.Module]:
    """Returns a non-linearity given name and dimension"""
    if name == "irelu":
        return [nn.ReLU(inplace=True)]
    if name == "relu":
        return [nn.ReLU()]
    if name == "ntanh":
        return [ParallelLayerNorm(normalized_shape=[dim], n_parallel=n_parallel), nn.Tanh()]
    if name == "layernorm":
        return [ParallelLayerNorm([dim], n_parallel=n_parallel)]
    if name == "tanh":
        return [nn.Tanh()]
    raise ValueError(f"Unknown non-linearity {name}")


def parallel_mlp(*layers: tp.Sequence[tp.Union[int, str]], n_parallel: int = 2) -> nn.Sequential:
    assert len(layers) >= 2
    sequence: tp.List[nn.Module] = []
    assert isinstance(layers[0], int), "First input must provide the dimension"
    prev_dim: int = layers[0]
    for layer in layers[1:]:
        if isinstance(layer, str):
            sequence.extend(_parallel_nl(layer, prev_dim, n_parallel=n_parallel))
        else:
            assert isinstance(layer, int)
            sequence.append(DenseParallel(prev_dim, layer, n_parallel=n_parallel))
            prev_dim = layer
    return nn.Sequential(*sequence)


class PhiMap(nn.Module):
    def __init__(self, goal_dim, z_dim, hidden_dim, hidden_layers: int = 2, norm=True, batch_norm=False) -> None:
        super().__init__()
        seq = [goal_dim, hidden_dim, "ntanh"]
        for _ in range(hidden_layers-1):
            seq += [hidden_dim, "relu"]
        seq += [z_dim]
        if norm:
            seq += ["norm"]
        if batch_norm:
            seq += ["bnorm"]
        self.net = mlp(*seq)

    def forward(self, x):
        return self.net(x)



def build_embedding(input_dim, hidden_dim, hidden_layers, num_parallel=1):
    assert hidden_layers >= 2, "must have at least 2 embedding layers"
    seq = [input_dim, hidden_dim, "ntanh"]
    for _ in range(hidden_layers-2):
        seq += [hidden_dim, "relu"]
    seq += [hidden_dim // 2, "relu"]
    if num_parallel == 1:
        return mlp(*seq)
    return parallel_mlp(*seq, n_parallel=num_parallel)



def build_embedding_relu(input_dim, hidden_dim, hidden_layers, num_parallel=1):
    assert hidden_layers >= 2, "must have at least 2 embedding layers"
    seq = [input_dim, hidden_dim, "relu"]
    for _ in range(hidden_layers-2):
        seq += [hidden_dim, "relu"]
    seq += [hidden_dim // 2, "relu"]
    if num_parallel == 1:
        return mlp(*seq)
    return parallel_mlp(*seq, n_parallel=num_parallel)


class PsiMap(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, hidden_dim, hidden_layers: int = 1,
                 embedding_layers: int = 2, num_parallel: int = 2, output_dim=None) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.num_parallel = num_parallel
        self.hidden_dim = hidden_dim

        self.embed_z = build_embedding(obs_dim + z_dim, hidden_dim, embedding_layers, num_parallel)
        self.embed_sa = build_embedding(obs_dim + action_dim, hidden_dim, embedding_layers, num_parallel)

        seq = [hidden_dim] + [hidden_dim, "relu"] * hidden_layers + [output_dim if output_dim else z_dim]
        if num_parallel == 1:
            self.Fs = mlp(*seq)
        else:
            self.Fs = parallel_mlp(*seq, n_parallel=num_parallel)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, action: tp.Optional[torch.Tensor] = None):
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            if action is not None:
                action = action.expand(self.num_parallel, -1, -1)
        # print(obs.shape)
        # print(z.shape)
        # print(torch.cat([obs, z], dim=-1).shape)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1)) # num_parallel x bs x h_dim // 2
        if action is not None:
            sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1)) # num_parallel x bs x h_dim // 2
        else:
            sa_embedding = self.embed_sa(obs) # num_parallel x bs x h_dim // 2
        return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))

class SimpleActor(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, hidden_dim, hidden_layers: int = 1,
                 embedding_layers: int = 2) -> None:
        super().__init__()

        # self.embed_z = build_embedding(obs_dim + z_dim, hidden_dim, embedding_layers)
        # self.embed_s = build_embedding(obs_dim, hidden_dim, embedding_layers)
        seq = [obs_dim+z_dim,hidden_dim,"ntanh"]+[hidden_dim,"relu"]+[hidden_dim,"relu"] + [hidden_dim, "relu"] * hidden_layers + [action_dim]
        # seq = [obs_dim + z_dim,hidden_dim,"ntanh"]+ [hidden_dim,] + [hidden_dim, "relu"] * hidden_layers + [action_dim]
        self.policy = mlp(*seq)

    def forward(self, obs, z, std):
        # z_embedding = self.embed_z(torch.cat([obs, z], dim=-1)) # bs x h_dim // 2
        # s_embedding = self.embed_s(obs) # bs x h_dim // 2
        # embedding = torch.cat([s_embedding, z_embedding], dim=-1)
        embedding = torch.cat([obs,z], dim=-1)
        mu = torch.tanh(self.policy(embedding))
        std = torch.ones_like(mu) * std
        dist = TruncatedNormal(mu, std)
        return dist


class Actor(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, hidden_dim, hidden_layers: int = 1,
                 embedding_layers: int = 2) -> None:
        super().__init__()

        self.embed_z = build_embedding(obs_dim + z_dim, hidden_dim, embedding_layers)
        self.embed_s = build_embedding(obs_dim, hidden_dim, embedding_layers)

        seq = [hidden_dim] + [hidden_dim, "relu"] * hidden_layers + [action_dim]
        self.policy = mlp(*seq)

    def forward(self, obs, z, std):
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1)) # bs x h_dim // 2
        s_embedding = self.embed_s(obs) # bs x h_dim // 2
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)
        mu = torch.tanh(self.policy(embedding))
        std = torch.ones_like(mu) * std
        dist = TruncatedNormal(mu, std)
        return dist
