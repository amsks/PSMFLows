"""Network definitions for the measure agents. Modules only — losses live in agents/.

Code <-> symbols (docs/reference/psmflow-symbols.md holds the full map; PSM is
arXiv 2411.19418):

  PhiMap                 phi(x), the basis over future states
  PsiMap                 psi(s, index, u), free successor-feature head
  AffinePsiMap           psi(s, u, u') = A(s,u)^T w(u') + beta(s,u)
  NoiseConditionedActor  pi_eta(s, w, eps) -> u, the flow-BC one-step latent actor
  FlowVectorField        v_xi(s, u_t, t), the actor's conditional-flow-matching field
  TanhGaussianLatentActor / LogAlpha   the DSRL-style latent actor and SAC's log alpha

These intentionally do NOT reuse utils/networks.MLP: the PSM reference uses a specific
activation/norm sequence — `ntanh` (LayerNorm then tanh), `relu`, and a final
`Norm` = sqrt(d) * x / ||x|| — that must be reproduced exactly for numerical
equivalence. flax LayerNorm uses epsilon=1e-5 to match torch's default.
"""

import math

import flax.linen as nn
import jax
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
    """Basis phi(x) -> R^z_dim: Dense, ntanh, [Dense, relu]*(L-1), Dense, [norm]."""

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
    """One (non-ensembled) successor-feature tower.

    Supports embedding_layers=2, hidden_layers=1 (the reference defaults). Submodules are
    named explicitly so the torch -> flax weight mapping is unambiguous.
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


class PsiMap(nn.Module):
    """Free psi(s, index, u) -> [num_parallel, B, output_dim].

    The policy coordinate w(u') is absorbed into the network (Rem. `tradeoff`'s "free
    psi"): the index slot is one more conditioning input, not a factor of the head.
    """

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
    output.
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


class TanhGaussianLatentActor(nn.Module):
    """DSRL-SAC's noise policy over flow latents.

    pi(u | s, w) = u_clip * tanh(mu(s, w) + exp(log_std(s, w)) * eps),  eps ~ N(0, I):
    a diagonal Gaussian in a pre-squash space, tanh-squashed, then rescaled into the
    latent box. `log_std` is clamped to SB3's [-20, 2] so a collapsing or exploding scale
    cannot NaN the log-prob.

    The trunk is `NoiseConditionedActor`'s minus the noise input -- the randomness is the
    reparameterisation now -- so the two heads are comparable at the same widths.

    Returns the PRE-SQUASH (mu, log_std); `tanh_gaussian_sample` does the squash, the
    scaling and the log-prob correction, so a caller that only wants the mode can take
    `u_clip * tanh(mu)` without paying for a draw.
    """

    action_dim: int
    hidden_dim: int = 512
    hidden_layers: int = 2
    embedding_layers: int = 2
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    #: `prior_init=True` starts the policy AT the flow's prior instead of outside it.
    #: With orthogonal init and log_std ~ 0 the step-0 latent has per-dim std
    #: scale * std(tanh N(0,1)) ~ 1.88 at scale=3, against the prior's 1.0, so the actor
    #: begins on decodes the flow was never fitted on and the first thousands of steps
    #: climb a critic evaluated off-manifold. Zeroing the mu head and biasing log_std to
    #: log(prior_init_std) starts it inside the support. Default False reproduces every
    #: run before 2026-09-08.
    prior_init: bool = False
    prior_init_std: float = 0.3
    #: LayerNorm after every hidden ReLU of the trunk. DSRL's offline OGBench configs use
    #: it on every hidden layer of both actor and critic; the embeddings already carry one
    #: (`_simple_embedding`), this covers the `hidden_layers` stack after them. Default
    #: False reproduces every dsrl_sac run before 2026-09-08.
    layer_norm: bool = False

    @nn.compact
    def __call__(self, obs, z):
        z_embedding = _simple_embedding(jnp.concatenate([obs, z], -1),
                                        self.hidden_dim, self.embedding_layers)
        s_embedding = _simple_embedding(obs, self.hidden_dim, self.embedding_layers)
        h = jnp.concatenate([s_embedding, z_embedding], -1)
        for _ in range(self.hidden_layers):
            h = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(h))
            if self.layer_norm:
                h = nn.LayerNorm(epsilon=1e-5)(h)
        if self.prior_init:
            mu = nn.Dense(self.action_dim, kernel_init=nn.initializers.zeros)(h)
            log_std = nn.Dense(
                self.action_dim, kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.constant(math.log(self.prior_init_std)))(h)
        else:
            mu = nn.Dense(self.action_dim, kernel_init=_ORTH1)(h)
            log_std = nn.Dense(self.action_dim, kernel_init=_ORTH1)(h)
        return mu, jnp.clip(log_std, self.log_std_min, self.log_std_max)


def tanh_gaussian_sample(mu, log_std, noise, scale):
    """Reparameterised draw from the tanh-Gaussian, plus its log-density.

        log pi = sum_i [ log N(pre_i; mu_i, sigma_i) - log(1 - tanh(pre_i)^2 + 1e-6) ]

    taken in the SQUASHED-but-UNSCALED space [-1, 1], where SB3 computes it. The
    `d_a * log(scale)` Jacobian of the box rescale is deliberately omitted: it is a
    constant shift, but the entropy target alpha is tuned against is stated in that same
    unscaled space, so adding it would move the target by d_a * log(u_clip).
    """
    pre = mu + jnp.exp(log_std) * noise
    tanh_pre = jnp.tanh(pre)
    logp = (-0.5 * noise ** 2 - log_std - 0.5 * jnp.log(2.0 * jnp.pi)).sum(-1)
    logp = logp - jnp.log(1.0 - tanh_pre ** 2 + 1e-6).sum(-1)
    return scale * tanh_pre, logp


class LogAlpha(nn.Module):
    """SAC's entropy coefficient as one scalar parameter, so it can live in a TrainState.

    Held as log alpha, so the Adam step is multiplicative in alpha and the coefficient
    cannot go negative.
    """

    init_value: float = 0.0

    @nn.compact
    def __call__(self):
        return self.param("log_alpha",
                          lambda _: jnp.asarray(self.init_value, jnp.float32))


class FlowVectorField(nn.Module):
    """Flow velocity v(obs, u_t, t): Linear->GELU, (L-1)x[Linear->GELU], Linear(action_dim).

    GELU activations, orthogonal init, no LayerNorm.
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
# Affine psi (Prop. `bilinear`): psi(s,u,u') = A(s,u)^T w(u') + beta(s,u).
# ---------------------------------------------------------------------------

class _AffinePsiTower(nn.Module):
    """One (non-ensembled) (s,u)-side tower emitting A(s,u) and beta(s,u).

    The trunk is `_PsiTower`'s (s,a) branch -- Dense(h) -> LayerNorm -> tanh ->
    Dense(h//2) -> relu -> Dense(h) -> relu -- with two heads instead of one and no
    z-branch: under Assumption `affine` the basis and the bias are exactly the objects
    that do not depend on the policy index, so nothing about u' may reach them.
    """

    hidden_dim: int
    output_dim: int
    w_dim: int
    embedding_layers: int = 2
    hidden_layers: int = 1

    def setup(self):
        assert self.embedding_layers == 2 and self.hidden_layers == 1, \
            "only embedding_layers=2, hidden_layers=1 supported (reference default)"
        h = self.hidden_dim
        self.embed_sa_0 = nn.Dense(h, kernel_init=_ORTH_RELU)
        self.embed_sa_ln = nn.LayerNorm(epsilon=1e-5)
        self.embed_sa_3 = nn.Dense(h // 2, kernel_init=_ORTH_RELU)
        self.fs_0 = nn.Dense(h, kernel_init=_ORTH_RELU)
        # A is the only wide object: (hidden_dim -> z_dim * w_dim). Orthogonal gain 1, as
        # the reference uses for every output head.
        self.a_out = nn.Dense(self.output_dim * self.w_dim, kernel_init=_ORTH1)
        self.beta_out = nn.Dense(self.output_dim, kernel_init=_ORTH1)

    def __call__(self, obs, action):
        se = nn.relu(self.embed_sa_3(jnp.tanh(self.embed_sa_ln(
            self.embed_sa_0(jnp.concatenate([obs, action], -1))))))
        x = nn.relu(self.fs_0(se))
        A = self.a_out(x).reshape(*x.shape[:-1], self.output_dim, self.w_dim)
        return A, self.beta_out(x)


class AffinePsiMap(nn.Module):
    """psi(s, u, u') = A(s,u)^T w(u') + beta(s,u), Prop. `bilinear` made explicit.

    A drop-in for `PsiMap`: the same `(obs, index, u)` signature and the same
    [num_parallel, B, output_dim] return. The `index` slot must carry the policy latent u'
    (`policy_index=latent`); the point of the head is that the policy enters psi only
    through the finite coordinate w^{u'}, which a z_dim task vector there would contradict.

    Assumption `affine` asserts that w^{u'} exists but gives no formula, so the encoder
    u' -> w(u') is a design choice: an MLP with PhiMap's shape, shared across the ensemble
    because w^{u'} is a property of the POLICY rather than of a critic member. The
    ensemble then disagrees only through (A, beta), which is what the pessimism term
    measures.

    `norm_w` puts w(u') on the unit sphere. (A, w) is identified only up to (cA, w/c), and
    fixing ||w|| = 1 pins it while keeping psi's scale independent of w_dim, so the head
    is comparable to the free PsiMap at the same hidden_dim and a collapsed encoder shows
    up as small pairwise distance at fixed radius rather than as a shrinking norm.
    """

    output_dim: int
    hidden_dim: int
    num_parallel: int = 2
    embedding_layers: int = 2
    hidden_layers: int = 1
    w_dim: int = 128
    encoder_hidden: int = 256
    encoder_layers: int = 2
    norm_w: bool = True

    def setup(self):
        self.w_enc = PhiMap(z_dim=self.w_dim, hidden_dim=self.encoder_hidden,
                            hidden_layers=self.encoder_layers, norm=False)
        self.tower = ensemblize(_AffinePsiTower, self.num_parallel, in_axes=None)(
            hidden_dim=self.hidden_dim, output_dim=self.output_dim, w_dim=self.w_dim,
            embedding_layers=self.embedding_layers, hidden_layers=self.hidden_layers,
            name="tower",
        )

    def encode_index(self, index):
        """w(u') in R^{w_dim}. Exposed so the agent can log encoder collapse directly."""
        w = self.w_enc(index)
        if self.norm_w:
            w = w / jnp.maximum(jnp.linalg.norm(w, axis=-1, keepdims=True), 1e-12)
        return w

    def sa_terms(self, obs, u):
        """A(s,u) in R^{P x B x z x d_w} and beta(s,u) in R^{P x B x z}.

        Takes no policy index: that is Assumption `affine`, and it is what the affineness
        test checks against.
        """
        return self.tower(obs, u)

    def __call__(self, obs, index, u):
        w = self.encode_index(index)
        A, beta = self.sa_terms(obs, u)              # (P, B, z, w_dim), (P, B, z)
        w = jnp.broadcast_to(w, (*A.shape[:-2], self.w_dim))
        return jnp.einsum("...zw,...w->...z", A, w) + beta
