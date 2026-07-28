"""Bespoke PSM networks, transcribed from the PyTorch reference psm_nets.py.

Code <-> paper (arXiv 2411.19418):
  PhiMap   -> phi_s(s+)      basis over future states (the learned proto basis)
  PsiMap   -> psi^pi(s,a)    successor-feature coefficients (task or codebook head)
  PSMActor -> pi(a|s,z)      TD3 mean actor conditioned on the task vector w
  NoiseConditionedActor / FlowVectorField -> flow-BC one-step actor + velocity field

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


# ---------------------------------------------------------------------------
# Affine (full) PSM networks — the M = Phi(s,a,x)·w + b factorization.
# Distinct from the bilinear PhiMap/PsiMap above; see agents/affine_psm.py.
# ---------------------------------------------------------------------------

class AffineMeasureNet(nn.Module):
    """Affine successor-measure net (RLU psm.py PSM). Shared trunk on concat[obs,action,x]
    with two heads: phi (basis, R^d) and b (offset, R^1). M(s,a,x) = phi(s,a,x)·w + b.

    `norm=True` L2-normalizes phi to ||phi||=sqrt(d) (like PhiMap / Factored-FB). RLU's
    affine PSM leaves phi RAW, which lets the measure/TD-target diverge and then collapse
    (basis->0, M->b) on long runs; normalization bounds the measure and makes a high ortho
    coef behave as a pure decorrelator (the proven bilinear-cube recipe). Normalization
    keeps M linear in w, so the constrained-LP inference is unaffected.
    """

    d_dim: int
    hidden_dim: int
    hidden_layers: int = 2
    norm: bool = True
    b_scale: float = 10.0   # tanh-bound on the affine offset b; <=0 disables (raw b)

    @nn.compact
    def __call__(self, obs, action, x):
        inp = jnp.concatenate([obs, action, x], -1)

        def path(out_dim, name):
            # RLU builds mlp_phi and mlp_b as two SEPARATE trunks over the same input
            # (psm.py:174-187) — no shared features — each 5 Linear deep (3 trunk + 2 head).
            h = inp
            for i in range(self.hidden_layers):
                h = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_ORTH_RELU,
                                     name=f"{name}_trunk_{i}")(h))
            h = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_ORTH_RELU, name=f"{name}_head_h")(h))
            return nn.Dense(out_dim, kernel_init=_ORTH1, name=f"{name}_out")(h)

        phi = path(self.d_dim, "phi")
        if self.norm:
            phi = psm_norm(phi)
        b = path(1, "b")
        if self.b_scale > 0:
            # Hard-bound the bias: raw b is an unconstrained degeneracy sink that explodes
            # (~+-3000) once phi is normalized. Bounding it to +-b_scale keeps M finite while
            # phi*w (in +-d) carries the informative part.
            b = self.b_scale * jnp.tanh(b)
        return phi, b


class FactoredAffineMeasureNet(nn.Module):
    """Affine measure with a FACTORIZED basis: Phi(s,a,x) = A(s,a) phi_x(x).

    This is Prop. 4.3 of the write-up (assumption A5): factorizing the basis reduces the
    general affine form to a bilinear one that is STILL LINEAR in w, so the constrained-LP
    `full` inference is unaffected. (Remark 4.4's warning applies to *free*-psi
    implementations, which drop affineness in w; this is not one — A and beta are network
    outputs, w is not.)

    Why it exists: the unfactorized AffineMeasureNet takes x INSIDE the trunk, so the B^2
    contrastive mesh costs B^2 network evaluations — measured 274 ms/step at B=512 and
    ~1.1 s at B=1024, i.e. ~6 days per 500k-step seed. Factorized, the mesh costs B evals
    per tower plus two matmuls, exactly like the reference's psi(s,z,a) @ phi(g)^T. That is
    what makes batch_size=1024 (the reference cube value) affordable.

        M(s,a,x) = Phi(s,a,x)·w + b(s,a,x)
                 = phi_x(x)·(A(s,a)^T w)  +  b_scale*tanh(beta(s,a)·phi_x(x))

    `mesh_terms` returns the per-side factors so the agent can build the (B,B) mesh with
    two matmuls; `__call__` gives the elementwise value for the actor / inference paths.
    Both compute the SAME function — the mesh form is an algebraic regrouping, not an
    approximation.

    Normalization follows the reference rather than the unfactored net: PhiMap normalizes
    the x-side basis phi(g) and leaves psi free, so here psm_norm is applied to phi_x (the
    measure-argument basis) and NOT to the product Phi. That is what keeps the mesh cheap
    (normalizing Phi would force materializing the (B,B,d) product) and it keeps the ortho
    regularizer's diagonal term inert, so ortho_coef=1000 stays a pure decorrelator —
    the property configs/agent/affine_psm.yaml relies on. Apply ortho to phi_x, not Phi.
    """

    d_dim: int
    hidden_dim: int
    k_dim: int = 0          # factorization rank; 0 => d_dim. Keep it WELL BELOW d_dim: the A
                            # head emits d_dim*k_dim values, so k_dim=d_dim=128 makes it a
                            # 1024x16384 layer (16.8M params, 79% of the net) against the
                            # reference psi head's 1024x128.
    hidden_layers: int = 3
    norm: bool = True
    b_scale: float = 10.0   # tanh-bound on the affine offset b; <=0 disables (raw b)

    @property
    def _k(self):
        return self.k_dim if self.k_dim > 0 else self.d_dim

    def _tower(self, name):
        """RLU's shape: `hidden_layers` trunk layers, then a head of (hidden, out).

        RLU builds mlp_phi and mlp_b as two COMPLETELY SEPARATE trunks over the same input
        (psm.py:174-187) — it does not share features between the measure and the offset.
        Each path is 5 Linear layers deep (3 trunk + 2 head), so the head carries one
        hidden layer of its own. We mirror both properties.
        """
        return ([nn.Dense(self.hidden_dim, kernel_init=_ORTH_RELU, name=f"{name}_trunk_{i}")
                 for i in range(self.hidden_layers)],
                nn.Dense(self.hidden_dim, kernel_init=_ORTH_RELU, name=f"{name}_head_h"))

    def setup(self):
        k = self._k
        # Four independent towers: {phi, b} x {(s,a) side, x side}. The phi/b separation is
        # RLU's; the (s,a)/x separation is what makes the B^2 mesh cost B evals.
        self.a_trunk, self.a_head_h = self._tower("a")
        self.a_out = nn.Dense(self.d_dim * k, kernel_init=_ORTH1, name="a_out")
        self.beta_trunk, self.beta_head_h = self._tower("beta")
        self.beta_out = nn.Dense(k, kernel_init=_ORTH1, name="beta_out")
        self.xphi_trunk, self.xphi_head_h = self._tower("xphi")
        self.xphi_out = nn.Dense(k, kernel_init=_ORTH1, name="xphi_out")
        self.xb_trunk, self.xb_head_h = self._tower("xb")
        self.xb_out = nn.Dense(k, kernel_init=_ORTH1, name="xb_out")

    @staticmethod
    def _run(trunk, head_h, out, h):
        for layer in trunk:
            h = nn.relu(layer(h))
        return out(nn.relu(head_h(h)))

    def sa(self, obs, action):
        """(s,a) side: A with shape (B, d, k) and beta with shape (B, k)."""
        h = jnp.concatenate([obs, action], -1)
        A = self._run(self.a_trunk, self.a_head_h, self.a_out, h)
        A = A.reshape(*h.shape[:-1], self.d_dim, self._k)
        beta = self._run(self.beta_trunk, self.beta_head_h, self.beta_out, h)
        return A, beta

    def x(self, x):
        """Measure-argument bases: phi_x(x) (sqrt(k)-normalized, cf. PhiMap) and phi_b(x).

        phi_b is a SEPARATE basis for the offset, mirroring RLU's independent b path; it is
        left unnormalized because b is tanh-bounded downstream instead.
        """
        px = self._run(self.xphi_trunk, self.xphi_head_h, self.xphi_out, x)
        if self.norm:
            px = psm_norm(px)
        pb = self._run(self.xb_trunk, self.xb_head_h, self.xb_out, x)
        return px, pb

    def mesh_terms(self, obs, action, x):
        """Factors for the (B,B) mesh: (A, beta) on the source side, (phi_x, phi_b) on the
        measure-argument side. M = phi_x·(A^T w) + bound(beta·phi_b)."""
        A, beta = self.sa(obs, action)
        px, pb = self.x(x)
        return A, beta, px, pb

    def _bound_b(self, raw):
        return self.b_scale * jnp.tanh(raw) if self.b_scale > 0 else raw

    def __call__(self, obs, action, x):
        A, beta, px, pb = self.mesh_terms(obs, action, x)
        phi = jnp.einsum("...dk,...k->...d", A, px)
        return phi, self._bound_b(jnp.sum(beta * pb, axis=-1, keepdims=True))


class LagrangeNet(nn.Module):
    """Learned Lagrange multiplier lam(s,a,x) >= 0 for dual gradient descent
    (RLU psm.py `lmult`, softplus head)."""

    hidden_dim: int
    hidden_layers: int = 2

    @nn.compact
    def __call__(self, obs, action, x):
        h = jnp.concatenate([obs, action, x], -1)         # concat the (state, action, measure-arg) inputs
        for _ in range(self.hidden_layers):               # hidden_layers x [Dense -> ReLU]
            h = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_ORTH_RELU)(h))
        return nn.softplus(nn.Dense(1, kernel_init=_ORTH1)(h))   # softplus keeps the multiplier >= 0


class WNet(nn.Module):
    """Task-coordinate net: binary codebook code z (max_log_seed bits) -> w in R^d
    (RLU psm.py `self.w`). During reward-free training every codebook policy pi_z gets a
    learnable task coordinate w(z), and the affine measure M = Phi·w(z) + b is fit for it.
    (Default Dense init kept deliberately — matches the trained affine_psm checkpoints.)"""

    d_dim: int
    hidden_dim: int
    hidden_layers: int = 3

    @nn.compact
    def __call__(self, z):
        h = z
        for _ in range(self.hidden_layers):               # hidden_layers x [Dense -> ReLU]
            h = nn.relu(nn.Dense(self.hidden_dim)(h))
        return nn.Dense(self.d_dim)(h)                     # linear head -> d-dim task coordinate
