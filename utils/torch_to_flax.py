"""Helpers to map PyTorch reference weights (from the fixture) into flax param
pytrees, so the JAX PSM networks can be checked for numerical equivalence.

torch Linear.weight is [out, in]; flax Dense.kernel is [in, out] -> transpose.
torch LayerNorm.{weight,bias} -> flax {scale,bias}.
"""

import jax.numpy as jnp


def dense_params(weight, bias):
    return {"kernel": jnp.asarray(weight.T, jnp.float64), "bias": jnp.asarray(bias, jnp.float64)}


def layernorm_params(scale, bias):
    return {"scale": jnp.asarray(scale, jnp.float64), "bias": jnp.asarray(bias, jnp.float64)}


def load_phi_params(fix, prefix="phi"):
    """PhiMap with hidden_layers=2: torch net = Linear0, LN1, Tanh2, Linear3, ReLU4, Linear5.
    -> flax Dense_0, LayerNorm_0, Dense_1, Dense_2."""
    g = lambda k: fix[f"w__{prefix}.net.{k}"]
    return {
        "Dense_0": dense_params(g("0.weight"), g("0.bias")),
        "LayerNorm_0": layernorm_params(g("1.weight"), g("1.bias")),
        "Dense_1": dense_params(g("3.weight"), g("3.bias")),
        "Dense_2": dense_params(g("5.weight"), g("5.bias")),
    }
