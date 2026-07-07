"""Bespoke PSM networks, transcribed from the PyTorch reference psm_nets.py.

These intentionally do NOT reuse utils/networks.MLP: the reference uses a specific
activation/norm sequence — `ntanh` (LayerNorm then tanh), `relu`, and a final
`Norm` = sqrt(d) * x / ||x|| — that must be reproduced exactly for numerical
equivalence. flax LayerNorm uses epsilon=1e-5 to match torch's default.
"""

import math

import jax
import flax.linen as nn
import jax.numpy as jnp

from utils.networks import ensemblize

# Reference weight_init (agents/psm/psm_nets.py:75-87, nn_models.py:61-77):
#   nn.Linear      -> orthogonal_(gain=1),                bias 0
#   DenseParallel  -> parallel_orthogonal_(gain=relu≈√2), bias 0
# flax Dense bias defaults to zeros already, so we only override kernel_init.
_ORTH1 = nn.initializers.orthogonal()                       # gain 1 (plain Linear)
_ORTH_RELU = nn.initializers.orthogonal(scale=math.sqrt(2.0))  # ReLU gain (ensembled towers)


def truncated_clamp(x, low=-1.0, high=1.0, eps=1e-6):
    """Straight-through clamp from the reference TruncatedNormal._clamp."""
    clamped = jnp.clip(x, low + eps, high - eps)
    return x - jax.lax.stop_gradient(x) + jax.lax.stop_gradient(clamped)


def truncated_sample(loc, scale, noise, clip=None, low=-1.0, high=1.0, eps=1e-6):
    """Reference TruncatedNormal.sample with externally supplied standard-normal noise."""
    e = noise * scale
    if clip is not None:
        e = jnp.clip(e, -clip, clip)
    return truncated_clamp(loc + e, low, high, eps)


def psm_norm(x):
    """Reference `Norm`/`_L2`: sqrt(dim) * x / ||x||_2 (torch F.normalize eps=1e-12)."""
    d = x.shape[-1]
    denom = jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    return jnp.sqrt(d) * x / denom


class PhiMap(nn.Module):
    """phi(goal) -> R^z_dim. Sequence: Dense, ntanh, [Dense, relu]*(L-1), Dense, [norm]."""

    z_dim: int
    hidden_dim: int
    hidden_layers: int = 2
    norm: bool = True

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(x)
        x = nn.LayerNorm(epsilon=1e-5)(x)
        x = jnp.tanh(x)
        for _ in range(self.hidden_layers - 1):
            x = nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(x)
            x = nn.relu(x)
        x = nn.Dense(self.z_dim, kernel_init=_ORTH1)(x)
        if self.norm:
            x = psm_norm(x)
        return x


class _PsiTower(nn.Module):
    """One (non-ensembled) PSM successor-feature tower, transcribed from PsiMap.

    Supports embedding_layers=2 and hidden_layers=1 (the reference defaults / the
    configs we use). Submodules are explicitly named so the torch->flax weight
    mapping is unambiguous.
    """

    hidden_dim: int
    output_dim: int
    embedding_layers: int = 2
    hidden_layers: int = 1

    def setup(self):
        assert self.embedding_layers == 2 and self.hidden_layers == 1, \
            "only embedding_layers=2, hidden_layers=1 supported (reference default)"
        h = self.hidden_dim
        # DenseParallel in the reference -> parallel_orthogonal_ with ReLU gain (√2).
        self.embed_z_0 = nn.Dense(h, kernel_init=_ORTH_RELU)
        self.embed_z_ln = nn.LayerNorm(epsilon=1e-5)
        self.embed_z_3 = nn.Dense(h // 2, kernel_init=_ORTH_RELU)
        self.embed_sa_0 = nn.Dense(h, kernel_init=_ORTH_RELU)
        self.embed_sa_ln = nn.LayerNorm(epsilon=1e-5)
        self.embed_sa_3 = nn.Dense(h // 2, kernel_init=_ORTH_RELU)
        self.fs_0 = nn.Dense(h, kernel_init=_ORTH_RELU)
        self.fs_2 = nn.Dense(self.output_dim, kernel_init=_ORTH_RELU)

    def __call__(self, obs, z, action):
        ze = nn.relu(self.embed_z_3(jnp.tanh(self.embed_z_ln(self.embed_z_0(jnp.concatenate([obs, z], -1))))))
        se = nn.relu(self.embed_sa_3(jnp.tanh(self.embed_sa_ln(self.embed_sa_0(jnp.concatenate([obs, action], -1))))))
        x = jnp.concatenate([se, ze], -1)
        x = nn.relu(self.fs_0(x))
        return self.fs_2(x)


class PSMActor(nn.Module):
    """TD3 actor (reference psm_nets.Actor). Returns the mean mu = tanh(policy(emb)).

    embeds are non-parallel; embedding_layers=2, hidden_layers=1 supported.
    """

    action_dim: int
    hidden_dim: int
    embedding_layers: int = 2
    hidden_layers: int = 1

    def setup(self):
        assert self.embedding_layers == 2 and self.hidden_layers == 1
        h = self.hidden_dim
        self.embed_z_0 = nn.Dense(h, kernel_init=_ORTH1)
        self.embed_z_ln = nn.LayerNorm(epsilon=1e-5)
        self.embed_z_3 = nn.Dense(h // 2, kernel_init=_ORTH1)
        self.embed_s_0 = nn.Dense(h, kernel_init=_ORTH1)
        self.embed_s_ln = nn.LayerNorm(epsilon=1e-5)
        self.embed_s_3 = nn.Dense(h // 2, kernel_init=_ORTH1)
        self.policy_0 = nn.Dense(h, kernel_init=_ORTH1)
        self.policy_2 = nn.Dense(self.action_dim, kernel_init=_ORTH1)

    def __call__(self, obs, z):
        ze = nn.relu(self.embed_z_3(jnp.tanh(self.embed_z_ln(self.embed_z_0(jnp.concatenate([obs, z], -1))))))
        se = nn.relu(self.embed_s_3(jnp.tanh(self.embed_s_ln(self.embed_s_0(obs)))))
        emb = jnp.concatenate([se, ze], -1)
        return jnp.tanh(self.policy_2(nn.relu(self.policy_0(emb))))


class PsiMap(nn.Module):
    """Ensembled successor-feature net -> [num_parallel, B, output_dim]."""

    output_dim: int
    hidden_dim: int
    num_parallel: int = 2
    embedding_layers: int = 2
    hidden_layers: int = 1

    @nn.compact
    def __call__(self, obs, z, action):
        tower = ensemblize(_PsiTower, self.num_parallel, in_axes=None)(
            hidden_dim=self.hidden_dim, output_dim=self.output_dim,
            embedding_layers=self.embedding_layers, hidden_layers=self.hidden_layers,
            name="tower",
        )
        return tower(obs, z, action)


def _simple_embedding(x, hidden_dim, embedding_layers):
    """Reference nn_models.simple_embedding (num_parallel=1): Linear->LayerNorm->Tanh,
    (embedding_layers-2)x[Linear->ReLU], Linear(hidden//2)->ReLU. Returns hidden//2."""
    assert embedding_layers >= 2, "must have at least 2 embedding layers"
    x = jnp.tanh(nn.LayerNorm(epsilon=1e-5)(nn.Dense(hidden_dim, kernel_init=_ORTH1)(x)))
    for _ in range(embedding_layers - 2):
        x = nn.relu(nn.Dense(hidden_dim, kernel_init=_ORTH1)(x))
    return nn.relu(nn.Dense(hidden_dim // 2, kernel_init=_ORTH1)(x))


class NoiseConditionedActor(nn.Module):
    """Faithful port of nn_models.NoiseConditionedActor (the FlowBC one-step actor head).

    a = tanh(policy(concat[ embed_s([obs,noise]), embed_z([obs,z,noise]) ])).
    Each embedding is Linear->LayerNorm->Tanh->...->Linear(h//2)->ReLU; the policy is
    `hidden_layers` x [Linear->ReLU] then Linear(action_dim). All-orthogonal init, tanh
    output. This REPLACES the previous flat GELU ActorVectorField reuse.
    """

    action_dim: int
    hidden_dim: int = 512
    hidden_layers: int = 2
    embedding_layers: int = 2

    @nn.compact
    def __call__(self, obs, z, noise):
        z_embedding = _simple_embedding(jnp.concatenate([obs, z, noise], -1),
                                        self.hidden_dim, self.embedding_layers)
        s_embedding = _simple_embedding(jnp.concatenate([obs, noise], -1),
                                        self.hidden_dim, self.embedding_layers)
        h = jnp.concatenate([s_embedding, z_embedding], -1)
        for _ in range(self.hidden_layers):
            h = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(h))
        return jnp.tanh(nn.Dense(self.action_dim, kernel_init=_ORTH1)(h))


class FlowVectorField(nn.Module):
    """Faithful port of nn_models.VectorField (SimpleVectorField): unconditional flow
    velocity v(obs, x_t, t). Linear->GELU, (hidden_layers-1)x[Linear->GELU], Linear(adim).
    GELU activations (matches reference), orthogonal init, no LayerNorm.
    """

    action_dim: int
    hidden_dim: int = 512
    hidden_layers: int = 4

    @nn.compact
    def __call__(self, obs, action, t):
        x = jnp.concatenate([obs, action, t], -1)
        # torch nn.GELU() is the exact (erf) gelu; flax nn.gelu defaults to the tanh approx.
        x = nn.gelu(nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(x), approximate=False)
        for _ in range(self.hidden_layers - 1):
            x = nn.gelu(nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(x), approximate=False)
        return nn.Dense(self.action_dim, kernel_init=_ORTH1)(x)
