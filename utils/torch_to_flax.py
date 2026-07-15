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


def edense_params(weight, bias):
    """Ensembled DenseParallel: weight is already [P,in,out] (no transpose);
    bias [P,1,out] -> [P,out]."""
    return {"kernel": jnp.asarray(weight, jnp.float64), "bias": jnp.asarray(bias[:, 0, :], jnp.float64)}


def eln_params(weight, bias):
    """ParallelLayerNorm weight/bias [P,1,dim] -> flax scale/bias [P,dim]."""
    return {"scale": jnp.asarray(weight[:, 0, :], jnp.float64), "bias": jnp.asarray(bias[:, 0, :], jnp.float64)}


def load_psi_params(fix, which):
    """which in {'sf_psi','psm_psi'}. Maps to the _PsiTower tree under 'tower'."""
    g = lambda k: fix[f"w__{which}.{k}"]
    tower = {
        "embed_z_0": edense_params(g("embed_z.0.weight"), g("embed_z.0.bias")),
        "embed_z_ln": eln_params(g("embed_z.1.weight"), g("embed_z.1.bias")),
        "embed_z_3": edense_params(g("embed_z.3.weight"), g("embed_z.3.bias")),
        "embed_sa_0": edense_params(g("embed_sa.0.weight"), g("embed_sa.0.bias")),
        "embed_sa_ln": eln_params(g("embed_sa.1.weight"), g("embed_sa.1.bias")),
        "embed_sa_3": edense_params(g("embed_sa.3.weight"), g("embed_sa.3.bias")),
        "fs_0": edense_params(g("Fs.0.weight"), g("Fs.0.bias")),
        "fs_2": edense_params(g("Fs.2.weight"), g("Fs.2.bias")),
    }
    return {"tower": tower}


def load_actor_params(fix, prefix="actor"):
    """PSMActor: non-parallel Linear embeds + policy. setup() names:
    embed_z_0/ln/3, embed_s_0/ln/3, policy_0/policy_2."""
    g = lambda k: fix[f"w__{prefix}.{k}"]
    return {
        "embed_z_0": dense_params(g("embed_z.0.weight"), g("embed_z.0.bias")),
        "embed_z_ln": layernorm_params(g("embed_z.1.weight"), g("embed_z.1.bias")),
        "embed_z_3": dense_params(g("embed_z.3.weight"), g("embed_z.3.bias")),
        "embed_s_0": dense_params(g("embed_s.0.weight"), g("embed_s.0.bias")),
        "embed_s_ln": layernorm_params(g("embed_s.1.weight"), g("embed_s.1.bias")),
        "embed_s_3": dense_params(g("embed_s.3.weight"), g("embed_s.3.bias")),
        "policy_0": dense_params(g("policy.0.weight"), g("policy.0.bias")),
        "policy_2": dense_params(g("policy.2.weight"), g("policy.2.bias")),
    }


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


# ── FB loaders ───────────────────────────────────────────────────────────────
# The FB reference module attribute names carry a leading underscore
# (_backward_map, _forward_map, _left_encoder, _actor, _actor_vf), so the fixture
# keys look like `w___backward_map.net.0.weight` (w__ + _backward_map).


def load_backward_params(fix, prefix="_backward_map"):
    """BackwardMap/left_encoder with hidden_layers=4 (= PhiMap). torch net Linears at
    indices 0,3,5,7,9 (LN at 1, Norm at 10 has no params) -> flax Dense_0..Dense_4."""
    g = lambda k: fix[f"w__{prefix}.net.{k}"]
    return {
        "Dense_0": dense_params(g("0.weight"), g("0.bias")),
        "LayerNorm_0": layernorm_params(g("1.weight"), g("1.bias")),
        "Dense_1": dense_params(g("3.weight"), g("3.bias")),
        "Dense_2": dense_params(g("5.weight"), g("5.bias")),
        "Dense_3": dense_params(g("7.weight"), g("7.bias")),
        "Dense_4": dense_params(g("9.weight"), g("9.bias")),
    }


def load_left_encoder_params(fix):
    return load_backward_params(fix, prefix="_left_encoder")


def load_forward_params(fix, prefix="_forward_map"):
    """ForwardMap (ensembled): DenseParallel embeds + Fs trunk. hidden_layers=2 =>
    torch Fs Linears at 0,2,4 -> flax fs_0,fs_1,fs_2 under the 'tower' vmap scope."""
    g = lambda k: fix[f"w__{prefix}.{k}"]
    tower = {
        "embed_z_0": edense_params(g("embed_z.0.weight"), g("embed_z.0.bias")),
        "embed_z_ln": eln_params(g("embed_z.1.weight"), g("embed_z.1.bias")),
        "embed_z_3": edense_params(g("embed_z.3.weight"), g("embed_z.3.bias")),
        "embed_sa_0": edense_params(g("embed_sa.0.weight"), g("embed_sa.0.bias")),
        "embed_sa_ln": eln_params(g("embed_sa.1.weight"), g("embed_sa.1.bias")),
        "embed_sa_3": edense_params(g("embed_sa.3.weight"), g("embed_sa.3.bias")),
        "fs_0": edense_params(g("Fs.0.weight"), g("Fs.0.bias")),
        "fs_1": edense_params(g("Fs.2.weight"), g("Fs.2.bias")),
        "fs_2": edense_params(g("Fs.4.weight"), g("Fs.4.bias")),
    }
    return {"tower": tower}


def load_flow_actor(fix, prefix="_actor"):
    """NoiseConditionedActor: two simple_embeddings + policy. compact auto-names:
    embed_z -> Dense_0/LayerNorm_0/Dense_1, embed_s -> Dense_2/LayerNorm_1/Dense_3,
    policy(hidden_layers=2) -> Dense_4/Dense_5/Dense_6."""
    g = lambda k: fix[f"w__{prefix}.{k}"]
    return {
        "Dense_0": dense_params(g("embed_z.0.weight"), g("embed_z.0.bias")),
        "LayerNorm_0": layernorm_params(g("embed_z.1.weight"), g("embed_z.1.bias")),
        "Dense_1": dense_params(g("embed_z.3.weight"), g("embed_z.3.bias")),
        "Dense_2": dense_params(g("embed_s.0.weight"), g("embed_s.0.bias")),
        "LayerNorm_1": layernorm_params(g("embed_s.1.weight"), g("embed_s.1.bias")),
        "Dense_3": dense_params(g("embed_s.3.weight"), g("embed_s.3.bias")),
        "Dense_4": dense_params(g("policy.0.weight"), g("policy.0.bias")),
        "Dense_5": dense_params(g("policy.2.weight"), g("policy.2.bias")),
        "Dense_6": dense_params(g("policy.4.weight"), g("policy.4.bias")),
    }


def load_flow_vf(fix, prefix="_actor_vf"):
    """VectorField (hidden_layers=4): torch net Linears at 0,2,4,6,8 -> Dense_0..Dense_4."""
    g = lambda k: fix[f"w__{prefix}.net.{k}"]
    return {
        "Dense_0": dense_params(g("0.weight"), g("0.bias")),
        "Dense_1": dense_params(g("2.weight"), g("2.bias")),
        "Dense_2": dense_params(g("4.weight"), g("4.bias")),
        "Dense_3": dense_params(g("6.weight"), g("6.bias")),
        "Dense_4": dense_params(g("8.weight"), g("8.bias")),
    }
