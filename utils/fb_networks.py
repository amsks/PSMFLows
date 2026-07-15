"""FB networks (Forward/Backward maps), transcribed from the PyTorch reference
nn_models.py (ForwardMap, BackwardMap) for bit-exact parity.

Reuses the shared primitives from utils/psm_networks (orthogonal inits, ensemblize):
- BackwardMap is structurally identical to PhiMap (Dense->LN->tanh,
  (hidden_layers-1)x[Dense->relu], Dense(z_dim), optional Norm) so we alias it.
- ForwardMap mirrors PsiMap but generalizes the trunk (`Fs`) to `hidden_layers`
  linears + an output linear (cube uses hidden_layers=2 => 3 trunk linears), and
  embeds [left_enc(obs), z] and [left_enc(obs), action] (continuous z, no binary seed).
"""

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import ensemblize
from utils.psm_networks import _ORTH1, _ORTH_RELU, PhiMap, _simple_embedding

# B(next_obs) is a PhiMap: Dense->LayerNorm->tanh, (hidden_layers-1)x[Dense->relu],
# Dense(z_dim), then Norm (sqrt(d) L2) when norm=True. left_encoder is the same module.
BackwardMap = PhiMap


class _ForwardTower(nn.Module):
    """One (non-ensembled) forward-map tower. embed_z over [obs_feat, z], embed_sa
    over [obs_feat, action]; concat -> `hidden_layers` x [Dense->relu] -> Dense(z_dim).
    Submodules explicitly named so the torch->flax mapping is unambiguous (mirrors
    _PsiTower). Supports embedding_layers=2 (the reference default)."""

    z_dim: int
    hidden_dim: int
    hidden_layers: int = 2
    embedding_layers: int = 2

    def setup(self):
        assert self.embedding_layers == 2, "only embedding_layers=2 supported (reference default)"
        h = self.hidden_dim
        self.embed_z_0 = nn.Dense(h, kernel_init=_ORTH_RELU)
        self.embed_z_ln = nn.LayerNorm(epsilon=1e-5)
        self.embed_z_3 = nn.Dense(h // 2, kernel_init=_ORTH_RELU)
        self.embed_sa_0 = nn.Dense(h, kernel_init=_ORTH_RELU)
        self.embed_sa_ln = nn.LayerNorm(epsilon=1e-5)
        self.embed_sa_3 = nn.Dense(h // 2, kernel_init=_ORTH_RELU)
        # `hidden_layers` trunk linears (each relu'd) + a final output linear (no relu).
        self.fs = [nn.Dense(h, kernel_init=_ORTH_RELU) for _ in range(self.hidden_layers)] \
            + [nn.Dense(self.z_dim, kernel_init=_ORTH_RELU)]

    def __call__(self, obs_feat, z, action):
        ze = nn.relu(self.embed_z_3(jnp.tanh(self.embed_z_ln(
            self.embed_z_0(jnp.concatenate([obs_feat, z], -1))))))
        sa = nn.relu(self.embed_sa_3(jnp.tanh(self.embed_sa_ln(
            self.embed_sa_0(jnp.concatenate([obs_feat, action], -1))))))
        x = jnp.concatenate([sa, ze], -1)
        for layer in self.fs[:-1]:
            x = nn.relu(layer(x))
        return self.fs[-1](x)


class FBTd3Actor(nn.Module):
    """TD3 actor (reference nn_models.Actor): mu = tanh(policy(concat[embed_s(obs),
    embed_z([obs,z])])). Generalizes over hidden_layers (cube uses 2), unlike PSM's
    PSMActor which is fixed at hidden_layers=1."""

    action_dim: int
    hidden_dim: int
    hidden_layers: int = 2
    embedding_layers: int = 2

    @nn.compact
    def __call__(self, obs, z):
        ze = _simple_embedding(jnp.concatenate([obs, z], -1), self.hidden_dim, self.embedding_layers)
        se = _simple_embedding(obs, self.hidden_dim, self.embedding_layers)
        h = jnp.concatenate([se, ze], -1)
        for _ in range(self.hidden_layers):
            h = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(h))
        return jnp.tanh(nn.Dense(self.action_dim, kernel_init=_ORTH1)(h))


class ForwardMap(nn.Module):
    """Ensembled forward map F(left_enc(obs), z, action) -> [num_parallel, B, z_dim]."""

    z_dim: int
    hidden_dim: int
    num_parallel: int = 2
    hidden_layers: int = 2
    embedding_layers: int = 2

    @nn.compact
    def __call__(self, obs_feat, z, action):
        tower = ensemblize(_ForwardTower, self.num_parallel, in_axes=None)(
            z_dim=self.z_dim, hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers, embedding_layers=self.embedding_layers,
            name="tower",
        )
        return tower(obs_feat, z, action)
