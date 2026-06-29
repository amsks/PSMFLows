"""JAX/Flax FlowBC actor nets for vendored OGBench CRL, matching FB's PyTorch
VectorField (nn_models.py:416) and NoiseConditionedActor (nn_models.py:280).

Pure nets (no OGBench imports); the GC* wrappers hold optional encoders so the
agent can track encoder params via the OGBench ModuleDict. GELU is the exact
(erf) form and LayerNorm eps=1e-5 to match torch defaults. Imported by
crl_flowbc.py inside the OGBench child (jax env) and by the parity tests.
"""
import flax.linen as nn
import jax
import jax.numpy as jnp


class VectorField(nn.Module):
    """Unconditional (obs-only) velocity field: concat[obs, action, t] -> velocity.
    Mirrors FB VectorField: [Dense->GELU] + [Dense->GELU]*(L-1) + Dense(act)."""

    hidden_dim: int = 512
    hidden_layers: int = 4

    @nn.compact
    def __call__(self, obs, action, t):
        out_dim = action.shape[-1]
        x = jnp.concatenate([obs, action, t], axis=-1)
        x = nn.gelu(nn.Dense(self.hidden_dim, name="l0")(x), approximate=False)
        for i in range(self.hidden_layers - 1):
            x = nn.gelu(nn.Dense(self.hidden_dim, name=f"l{i + 1}")(x), approximate=False)
        return nn.Dense(out_dim, name="out")(x)


class NoiseConditionedActor(nn.Module):
    """One-shot actor: tanh(policy(concat[embed_s(obs,noise), embed_z(obs,z,noise)])).
    Mirrors FB NoiseConditionedActor + simple_embedding (embedding_layers>=2)."""

    hidden_dim: int = 512
    hidden_layers: int = 2
    embedding_layers: int = 2

    def _embed(self, x, prefix):
        h = self.hidden_dim
        x = nn.Dense(h, name=f"{prefix}_l0")(x)
        x = nn.LayerNorm(epsilon=1e-5, name=f"{prefix}_ln")(x)
        x = jnp.tanh(x)
        for i in range(self.embedding_layers - 2):
            x = nn.relu(nn.Dense(h, name=f"{prefix}_l{i + 1}")(x))
        return nn.relu(nn.Dense(h // 2, name=f"{prefix}_lout")(x))

    @nn.compact
    def __call__(self, obs, z, noise):
        out_dim = noise.shape[-1]
        z_emb = self._embed(jnp.concatenate([obs, z, noise], axis=-1), "embed_z")
        s_emb = self._embed(jnp.concatenate([obs, noise], axis=-1), "embed_s")
        x = jnp.concatenate([s_emb, z_emb], axis=-1)  # FB order: s then z
        for i in range(self.hidden_layers):
            x = nn.relu(nn.Dense(self.hidden_dim, name=f"policy_l{i}")(x))
        return jnp.tanh(nn.Dense(out_dim, name="policy_out")(x))


class GCVectorField(nn.Module):
    """VectorField with an optional obs encoder (for pixel obs)."""

    hidden_dim: int = 512
    hidden_layers: int = 4
    obs_encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations, actions, t):
        if self.obs_encoder is not None:
            observations = self.obs_encoder(observations)
        return VectorField(self.hidden_dim, self.hidden_layers)(observations, actions, t)


class GCNoiseActor(nn.Module):
    """NoiseConditionedActor with optional obs and goal encoders."""

    hidden_dim: int = 512
    hidden_layers: int = 2
    embedding_layers: int = 2
    obs_encoder: nn.Module = None
    goal_encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations, goals, noise):
        if self.obs_encoder is not None:
            observations = self.obs_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)
        return NoiseConditionedActor(
            self.hidden_dim, self.hidden_layers, self.embedding_layers
        )(observations, goals, noise)


def compute_flow_actions(vf, params, obs, noise, flow_steps):
    """Euler ODE rollout of `vf` (matches FB agent.py:118-125). `vf` is a
    VectorField module; `params` its param tree (already obs-encoded inputs)."""
    actions = noise
    for i in range(flow_steps):
        t = jnp.ones((noise.shape[0], 1)) * (i / flow_steps)
        vels = vf.apply({"params": params}, obs, actions, t)
        actions = actions + vels / flow_steps
    return jnp.clip(actions, -1.0, 1.0)
