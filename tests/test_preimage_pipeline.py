"""Task 2: preimage pipeline — point preimage, health scalars, and dataset point mode.

The mixture preimage (Task 1 / WP1) indexes a policy by a DISTRIBUTION over latents.
The point preimage is the exact backward-ODE solution, which the point-vs-mixture
ablation (`use_point_preimage`) trains against instead. `preimage_roundtrip` and
`preimage_ess` are the per-transition health scalars that let a training run tell
whether its inversion was trustworthy without re-running diagnostic D3.
"""
import jax
import ml_collections
import numpy as np
import pytest

from agents.fql import FQLAgent, get_config
from utils.datasets import Dataset
from utils.flow_inversion import (
    PREIMAGE_VALID_KEY,
    augment_dataset_with_point_preimage,
    augment_dataset_with_preimage_distribution,
    compute_preimage_validity,
    load_augmented_dataset,
    repair_invalid_preimages,
    save_augmented_dataset,
)

N, OBS, ACT = 12, 4, 2


@pytest.fixture(autouse=True)
def _force_float32():
    """The BC-flow inversion scan is not x64-safe; see tests/test_flow_inversion.py."""
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


def _dataset():
    rng = np.random.default_rng(0)
    return dict(
        observations=rng.standard_normal((N, OBS)).astype(np.float32),
        actions=np.clip(rng.standard_normal((N, ACT)), -1, 1).astype(np.float32),
        next_observations=rng.standard_normal((N, OBS)).astype(np.float32),
        terminals=np.zeros((N,), np.float32),
    )


def _agent():
    cfg = get_config()
    cfg["actor_hidden_dims"] = (32, 32)
    cfg["value_hidden_dims"] = (32, 32)
    cfg["flow_steps"] = 3
    ds = _dataset()
    return FQLAgent.create(0, ds["observations"][:1], ds["actions"][:1], cfg)


def _inv_cfg():
    return ml_collections.ConfigDict(
        dict(num_clusters=2, alpha=1.0, num_samples=6, n_steps=1, n_initial_steps=2,
             batch_size=8, seed=0)
    )


def test_point_and_health_arrays(tmp_path):
    agent, ds = _agent(), _dataset()
    aug = augment_dataset_with_preimage_distribution(agent, ds, _inv_cfg())
    aug = augment_dataset_with_point_preimage(agent, aug, _inv_cfg())
    assert aug["noise_preimage_point"].shape == (N, ACT)
    assert aug["preimage_roundtrip"].shape == (N,)
    assert aug["preimage_ess"].shape == (N,)
    assert np.all(np.isfinite(aug["noise_preimage_point"]))

    path = str(tmp_path / "pre.npz")
    save_augmented_dataset(path, aug)
    loaded = load_augmented_dataset(path)
    assert set(aug) == set(loaded)


def test_dataset_point_mode(tmp_path):
    agent, ds = _agent(), _dataset()
    aug = augment_dataset_with_preimage_distribution(agent, ds, _inv_cfg())
    aug = augment_dataset_with_point_preimage(agent, aug, _inv_cfg())

    d = Dataset.create(**aug)
    d.return_preimage_noise = True
    d.preimage_point_mode = True
    idxs = np.arange(5)
    batch = d.sample(5, idxs=idxs)
    np.testing.assert_array_equal(batch["noise_preimage"], aug["noise_preimage_point"][idxs])

    d2 = Dataset.create(**aug)
    d2.return_preimage_noise = True  # mixture mode (default)
    b2 = d2.sample(5, idxs=idxs)
    assert b2["noise_preimage"].shape == (5, ACT)


def test_ess_is_persisted_from_the_final_em_step():
    """`preimage_ess` must carry the EM's LAST iterate, not an intermediate one.

    The EM scan returns ess per step, shape (B, n_steps) after vmap. Storing the wrong
    axis element would silently report the health of a proposal that was then discarded.
    """
    agent, ds = _agent(), _dataset()
    cfg = _inv_cfg()
    cfg.n_steps = 4
    aug = augment_dataset_with_preimage_distribution(agent, ds, cfg)

    assert aug["preimage_ess"].shape == (N,)
    assert np.all(np.isfinite(aug["preimage_ess"]))
    # ESS is bounded by the number of importance samples drawn.
    assert np.all(aug["preimage_ess"] <= cfg.num_samples + 1e-4)
    assert np.all(aug["preimage_ess"] >= 1.0 - 1e-4)

    # It must equal the final scan step for the same seed, not any earlier one. This
    # reconstructs the loop's key threading (rng -> split -> per-row split) for the
    # first batch, so it pins the row alignment as well as the step index.
    ref = jax.jit(jax.vmap(
        lambda s, a, k: agent.compute_full_proposal_distribution_em(
            s, a, k, num_samples=cfg.num_samples, n_steps=cfg.n_steps,
            n_initial_steps=cfg.n_initial_steps, alpha=cfg.alpha, n_components=cfg.num_clusters,
        )[3]
    ))(
        ds["observations"][:8], ds["actions"][:8],
        jax.random.split(jax.random.split(jax.random.PRNGKey(cfg.seed))[1], 8),
    )
    np.testing.assert_allclose(
        aug["preimage_ess"][:8], np.asarray(ref)[:, -1], rtol=1e-5, atol=1e-5
    )


def _augmented_with_a_diverged_row(row=3):
    """A finite augmented dataset with one row poisoned the way a diverged inverse does."""
    agent, ds = _agent(), _dataset()
    aug = augment_dataset_with_preimage_distribution(agent, ds, _inv_cfg())
    aug = augment_dataset_with_point_preimage(agent, aug, _inv_cfg())
    aug['noise_preimage_mean'][row] = np.nan
    aug['noise_preimage_cov'][row] = np.nan
    aug['noise_preimage_point'][row] = np.nan
    aug['preimage_roundtrip'][row] = np.nan
    return aug, row


def test_validity_mask_flags_exactly_the_diverged_rows():
    """The flow inverse can diverge; the mask must find those rows and only those.

    `FQLAgent._get_preimage_and_jacobian` runs a fixed 5-sweep implicit-Euler fixed point
    with no convergence check, and on the cube Stage-A checkpoint 13 of 1M transitions
    diverge to NaN, which then spreads to every product built on the preimage.
    """
    aug, row = _augmented_with_a_diverged_row()
    valid = compute_preimage_validity(aug)

    assert valid.shape == (N,) and valid.dtype == np.float32
    assert valid[row] == 0.0
    assert valid.sum() == N - 1, 'a finite row was flagged, or the diverged row was missed'


def test_repair_resets_diverged_rows_to_the_prior_and_marks_them():
    """A diverged row must become trainable without inventing a preimage for it.

    A NaN latent means we do not know which u produced the action, so the mixture arm gets
    the N(0, I) prior — the honest posterior under no information. Rows cannot be dropped:
    `Dataset.sample` pairs each transition with row idx+1, so deleting rows would silently
    re-pair transitions across the gap.
    """
    aug, row = _augmented_with_a_diverged_row()
    finite_elsewhere = {k: v.copy() for k, v in aug.items() if k.startswith('noise_preimage')}

    # 1 of this fixture's 12 rows is 8.3%, over the production ceiling; the ceiling itself
    # is exercised by test_repair_aborts_rather_than_masking_a_broken_inversion.
    out, valid = repair_invalid_preimages(aug, max_invalid_frac=0.5)

    assert valid[row] == 0.0 and out[PREIMAGE_VALID_KEY][row] == 0.0
    for key, arr in out.items():
        if key.startswith('noise_preimage') or key.startswith('preimage_'):
            assert np.all(np.isfinite(arr)), f'{key} still non-finite after repair'
    # the prior, exactly
    np.testing.assert_allclose(out['noise_preimage_mean'][row], 0.0)
    for k in range(out['noise_preimage_cov'].shape[1]):
        np.testing.assert_allclose(out['noise_preimage_cov'][row, k], np.eye(ACT), atol=0)
    # and every OTHER row is untouched
    for key, before in finite_elsewhere.items():
        others = [i for i in range(N) if i != row]
        np.testing.assert_array_equal(out[key][others], before[others])


def test_repair_aborts_rather_than_masking_a_broken_inversion():
    """Repair is only defensible for a divergence tail, so it must refuse at scale.

    13/1M rows is a tail. If a large fraction is non-finite the inversion itself is wrong,
    and silently substituting the prior for those rows would train on fabricated latents
    while every health scalar looked clean.
    """
    aug, _ = _augmented_with_a_diverged_row()
    aug['noise_preimage_mean'][:] = np.nan  # every row diverged

    with pytest.raises(AssertionError, match='above the'):
        repair_invalid_preimages(aug)


def test_repair_recomputes_validity_for_npz_written_before_the_key_existed():
    """The cube npz predates `preimage_valid`; loading it must still be safe."""
    aug, row = _augmented_with_a_diverged_row()
    assert PREIMAGE_VALID_KEY not in aug

    out, valid = repair_invalid_preimages(aug, max_invalid_frac=0.5)

    assert valid[row] == 0.0
    assert np.all(np.isfinite(out['noise_preimage_mean']))


def test_roundtrip_health_matches_a_direct_decode():
    """`preimage_roundtrip` must be the real ||G(s, E(s,a)) - a||, not a placeholder."""
    agent, ds = _agent(), _dataset()
    aug = augment_dataset_with_preimage_distribution(agent, ds, _inv_cfg())
    aug = augment_dataset_with_point_preimage(agent, aug, _inv_cfg())

    recon = np.asarray(agent.compute_flow_actions(
        aug["observations"], noises=aug["noise_preimage_point"]))
    expected = np.linalg.norm(recon - np.clip(aug["actions"], -1, 1), axis=-1)
    np.testing.assert_allclose(aug["preimage_roundtrip"], expected, rtol=1e-4, atol=1e-5)
