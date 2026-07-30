"""Validation + plumbing tests for the BC-flow inversion utilities.

Task 1 (round-trip) characterizes Claas's inversion methods on FQLAgent. The code
under test already exists, so these are validation/characterization tests: they may
pass on first run. Round-trip consistency is a property of the inverter vs. the
network and holds regardless of flow training quality, as long as the forward and
inverse maps use the SAME step discretization.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.fql import FQLAgent, get_config


@pytest.fixture(autouse=True)
def _force_float32():
    """Run inversion in its real operating regime (float32).

    The PSM equivalence tests enable jax_enable_x64 globally at import, but the BC-flow
    inversion code (agents/fql.py implicit-Euler scan) is not x64-safe — under x64 the time
    `t` is float64 while the carry stays float32, breaking the scan carry-type invariant.
    Normal training/inference is float32, so force it off here and restore the prior value.
    """
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


def _tiny_agent(obs_dim=4, act_dim=2, flow_steps=100):
    """Build a minimal real FQLAgent. `create` derives ob_dims/action_dim and
    `encoder` defaults to None, so we only override flow_steps for speed/accuracy."""
    cfg = get_config()
    cfg['flow_steps'] = flow_steps
    ex_obs = jnp.zeros((1, obs_dim))
    ex_act = jnp.zeros((1, act_dim))
    return FQLAgent.create(0, ex_obs, ex_act, cfg)


def test_roundtrip_recovers_action():
    flow_steps = 100
    agent = _tiny_agent(obs_dim=4, act_dim=2, flow_steps=flow_steps)
    obs = jax.random.normal(jax.random.PRNGKey(0), (8, 4))
    noise = jax.random.normal(jax.random.PRNGKey(1), (8, 2))

    actions = agent.compute_flow_actions(obs, noises=noise)  # forward (uses cfg flow_steps)
    # invert each (single-example methods -> vmap); match n_steps to the forward discretization
    preimage = jax.vmap(
        lambda s, a: agent._get_preimage_and_jacobian(s, a, flow_steps)[0]
    )(obs, actions)
    recon = agent.compute_flow_actions(obs, noises=preimage)  # forward again
    err = float(jnp.mean(jnp.linalg.norm(recon - actions, axis=-1)))
    assert err < 1e-2, f"round-trip L2 {err} too high"


def test_augment_populates_preimage_slots():
    """Task 2 (WP1): the dataset inversion pass fills the noise-preimage mixture slots."""
    from utils.flow_inversion import augment_dataset_with_preimage_distribution

    agent = _tiny_agent(obs_dim=4, act_dim=2)
    N = 16
    ds = {
        'observations': np.zeros((N, 4), np.float32),
        'actions': np.zeros((N, 2), np.float32),
    }
    cfg = {'num_clusters': 3, 'alpha': 1.0, 'num_samples': 30,
           'n_steps': 3, 'n_initial_steps': 10, 'batch_size': 8, 'seed': 0}
    out = augment_dataset_with_preimage_distribution(agent, ds, cfg)
    assert out['noise_preimage_mean'].shape == (N, 3, 2)
    assert out['noise_preimage_cov'].shape == (N, 3, 2, 2)
    assert out['noise_preimage_weights'].shape == (N, 3)
    assert np.all(np.isfinite(out['noise_preimage_mean']))


def test_save_load_roundtrips_augmented_dataset(tmp_path):
    """Task 2 (WP1): persistence helpers round-trip the augmented dataset."""
    from utils.flow_inversion import save_augmented_dataset, load_augmented_dataset

    ds = {
        'observations': np.ones((5, 4), np.float32),
        'actions': np.zeros((5, 2), np.float32),
        'noise_preimage_mean': np.full((5, 1, 2), 0.3, np.float32),
        'noise_preimage_cov': np.tile(np.eye(2, dtype=np.float32), (5, 1, 1, 1)),
        'noise_preimage_weights': np.ones((5, 1), np.float32),
    }
    path = str(tmp_path / "aug.npz")
    save_augmented_dataset(path, ds)
    back = load_augmented_dataset(path)
    assert set(back.keys()) == set(ds.keys())
    for k in ds:
        np.testing.assert_array_equal(back[k], ds[k])


def test_batch_exposes_u0_and_u0prime():
    """Task 3 (WP1): a sampled batch carries u_0 (noise_preimage) and u_0' (next_noise_preimage)."""
    from utils.datasets import Dataset
    from utils.flow_inversion import augment_dataset_with_preimage_distribution

    agent = _tiny_agent(obs_dim=4, act_dim=2)
    N = 32
    ds = {
        'observations': np.zeros((N, 4), np.float32),
        'actions': np.zeros((N, 2), np.float32),
        'terminals': np.zeros(N, np.float32),
        'next_observations': np.zeros((N, 4), np.float32),
        'rewards': np.zeros(N, np.float32),
        'masks': np.ones(N, np.float32),
    }
    cfg = {'num_clusters': 2, 'alpha': 1.0, 'num_samples': 20,
           'n_steps': 2, 'n_initial_steps': 10, 'batch_size': 16, 'seed': 0}
    ds = augment_dataset_with_preimage_distribution(agent, ds, cfg)

    d = Dataset.create(**ds)
    d.return_preimage_noise = True
    b = d.sample(8)
    assert b['noise_preimage'].shape == (8, 2)
    assert b['next_noise_preimage'].shape == (8, 2)
    assert np.all(np.isfinite(b['noise_preimage']))
    assert np.all(np.isfinite(b['next_noise_preimage']))


def test_em_posterior_ess_is_finite():
    """The EM preimage posterior must not produce NaN ESS.

    A component whose responsibility mass collapses divides its scatter by a near-zero
    n_k, giving a huge / non-PSD covariance; the next iteration's MVN log_prob then
    returns NaN. Measured at ~1% of real cube-single transitions before the spectrum
    floor in compute_full_proposal_distribution_em — which would corrupt ~10k latents
    over a 1M-transition precompute, silently.

    Many components + few samples is the regime that provokes the collapse.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from agents.fql import FQLAgent, get_config

    cfg = get_config()
    cfg["flow_steps"] = 10
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.standard_normal((16, 19)), jnp.float32)
    act = jnp.asarray(np.clip(rng.standard_normal((16, 5)), -1, 1), jnp.float32)
    agent = FQLAgent.create(0, obs[:1], act[:1], cfg)

    keys = jax.random.split(jax.random.PRNGKey(0), obs.shape[0])
    _, covs, _, ess = jax.vmap(
        lambda s, a, k: agent.compute_full_proposal_distribution_em(
            s, a, k, num_samples=24, n_steps=12, n_initial_steps=10, alpha=1.0, n_components=4
        )
    )(obs, act, keys)

    assert np.all(np.isfinite(np.asarray(ess))), "EM produced non-finite ESS"
    # every component covariance stays PSD (the property the floor enforces)
    eigs = np.linalg.eigvalsh(np.asarray(covs))
    assert np.all(eigs > -1e-8), f"non-PSD covariance: min eig {eigs.min()}"


def test_em_ess_survives_a_diverging_flow_sample(monkeypatch):
    """One proposal sample whose forward flow diverges must not poison its whole row.

    At flow_steps>=100 the BC flow genuinely blows up for proposal noises in the tail.
    Measured on the cube Stage-A checkpoint (flow_steps=100, EM step 0): the proposal
    covariance saturates at the 1.0 clip in `_get_predistribution_proposal`, so samples
    are drawn ~N(x_0, I) and a ~0.02% minority land at ||u||~6.8 (vs a mean of 3.1).
    From there the integrated trajectory runs 6.5 -> 2.3e10 -> inf -> NaN by step 95.
    Coarser discretizations under-resolve the blow-up and stay finite, which is why
    flow_steps 10 and 30 never showed it.

    The EM's guards floored log_q but never masked log_energy, and softmax over a vector
    containing a single NaN is NaN in EVERY position: one diverged sample turned the whole
    row's sample_weights NaN -> NaN ESS and NaN means/covs, which then fed the next
    iteration. Measured cascade: 17% of rows NaN at EM step 0, 69% at step 1, 100% by
    step 5.

    A diverged sample is simply a bad preimage candidate and must take weight zero.
    """
    cfg = get_config()
    cfg["flow_steps"] = 10
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.standard_normal((8, 19)), jnp.float32)
    act = jnp.asarray(np.clip(rng.standard_normal((8, 5)), -1, 1), jnp.float32)
    agent = FQLAgent.create(0, obs[:1], act[:1], cfg)

    # Random weights do not diverge, so inject the failure the trained flow produces:
    # one sample of the proposal batch integrates to a non-finite action.
    real_compute_flow_actions = FQLAgent.compute_flow_actions

    def diverging(self, observations, noises):
        actions = real_compute_flow_actions(self, observations, noises)
        return actions.at[0].set(jnp.nan)

    monkeypatch.setattr(FQLAgent, "compute_flow_actions", diverging)

    keys = jax.random.split(jax.random.PRNGKey(0), obs.shape[0])
    means, covs, weights, ess = jax.vmap(
        lambda s, a, k: agent.compute_full_proposal_distribution_em(
            s, a, k, num_samples=24, n_steps=6, n_initial_steps=10, alpha=1.0, n_components=3
        )
    )(obs, act, keys)

    assert np.all(np.isfinite(np.asarray(ess))), "a single diverged sample poisoned the ESS"
    assert np.all(np.isfinite(np.asarray(means))), "a single diverged sample poisoned the means"
    assert np.all(np.isfinite(np.asarray(covs))), "a single diverged sample poisoned the covs"
    assert np.all(np.isfinite(np.asarray(weights))), "a single diverged sample poisoned the weights"
    # The diverged sample must be EXCLUDED, not merely tolerated: with 24 samples and one
    # dropped, ESS can never exceed the 23 that remain.
    assert np.all(np.asarray(ess) <= 23.0 + 1e-4), "diverged sample still carries weight"


def test_ess_reports_zero_not_num_samples_when_every_sample_is_rejected(monkeypatch):
    """ESS must not report its BEST value on total failure.

    When no sample is usable, the EM falls back to uniform logits so softmax returns
    1/num_samples for every sample, and 1/sum(w^2) evaluates to exactly num_samples --
    the maximum ESS can take. Measured on the cube Stage-A checkpoint: all 13 of the 1M
    rows whose stored mixture is NaN reported ESS=100/100. D3 gates on mean ESS, so the
    gate read perfect health on precisely the rows where the inversion had collapsed.
    """
    cfg = get_config()
    cfg["flow_steps"] = 10
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.standard_normal((4, 19)), jnp.float32)
    act = jnp.asarray(np.clip(rng.standard_normal((4, 5)), -1, 1), jnp.float32)
    agent = FQLAgent.create(0, obs[:1], act[:1], cfg)

    # Every sample non-finite == the all-rejected case the fallback exists for.
    monkeypatch.setattr(
        FQLAgent, "compute_flow_actions",
        lambda self, observations, noises: jnp.full_like(jnp.asarray(noises), jnp.nan))

    keys = jax.random.split(jax.random.PRNGKey(0), obs.shape[0])
    num_samples = 24
    _, _, _, ess = jax.vmap(
        lambda s, a, k: agent.compute_full_proposal_distribution_em(
            s, a, k, num_samples=num_samples, n_steps=3, n_initial_steps=10,
            alpha=1.0, n_components=3
        )
    )(obs, act, keys)

    ess = np.asarray(ess)
    assert np.all(np.isfinite(ess)), "the failure signal itself must stay finite"
    np.testing.assert_allclose(ess, 0.0, atol=1e-6)
    assert not np.any(ess >= num_samples - 1e-4), (
        "a row with zero usable samples reported the maximum possible ESS")
