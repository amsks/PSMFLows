"""Guard against the XLA:GPU autotuner miscompiling the flow integration.

See utils/xla_guard.py for the full measurement: with autotuning on, a long unrolled
Euler loop over a TRAINED BC flow compiles to a kernel returning garbage saturated at
the +-1 clip — 2.0 away from the un-jitted body, the full width of the action range.

IMPORTANT — why the numerics check is opt-in. A randomly-initialised flow does NOT
reproduce the bug: measured at B in {64,256} x ob_dim in {19,28} x flow_steps=100 with
autotuning deliberately re-enabled, jitted and un-jitted agree to 3.5e-5. The trigger
needs the trained weights, so a self-contained test would pass vacuously and give false
assurance. Point it at a real checkpoint instead:

    PSMFLOWS_FLOW_CKPT=/var/local/amsks/exp/PSMFLows/bcflow_*/sd000_* \
    PSMFLOWS_FLOW_CKPT_EPOCH=500000 .venv/bin/python -m pytest tests/test_xla_autotune_guard.py

What always runs is the plumbing test below, which is where a regression would most
plausibly land: someone reordering imports so the guard no longer precedes jax.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.fql import FQLAgent, get_config


def test_guard_sets_flag_and_preserves_existing(monkeypatch):
    """The guard appends its flag without clobbering flags the caller already set."""
    import utils.xla_guard as g

    monkeypatch.delenv("PSMFLOWS_ALLOW_XLA_AUTOTUNE", raising=False)
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
    g.disable_xla_autotune()
    flags = os.environ["XLA_FLAGS"]
    assert "--xla_force_host_platform_device_count=2" in flags
    assert "--xla_gpu_autotune_level=0" in flags

    # an explicit caller-chosen level wins, and the opt-out is honoured
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=4")
    g.disable_xla_autotune()
    assert os.environ["XLA_FLAGS"] == "--xla_gpu_autotune_level=4"

    monkeypatch.setenv("PSMFLOWS_ALLOW_XLA_AUTOTUNE", "1")
    monkeypatch.setenv("XLA_FLAGS", "")
    g.disable_xla_autotune()
    assert os.environ["XLA_FLAGS"] == ""


def test_entry_points_import_guard_before_jax():
    """main.py and the GPU tools must import utils.xla_guard ahead of jax.

    The flag is read when XLA initialises; setting it after `import jax` is a silent
    no-op, so import ORDER is the whole contract.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in ("main.py", "tools/validate_flow_inversion.py", "tools/precompute_preimages.py"):
        src = open(os.path.join(root, rel)).read()
        guard = src.find("utils.xla_guard")
        assert guard != -1, f"{rel} does not import utils.xla_guard"
        for jax_import in ("\nimport jax", "\nimport hydra"):
            at = src.find(jax_import)
            if at != -1:
                assert guard < at, f"{rel} imports {jax_import.strip()} before utils.xla_guard"


@pytest.mark.skipif(
    not os.environ.get("PSMFLOWS_FLOW_CKPT"),
    reason="needs a TRAINED flow checkpoint; random weights do not reproduce the bug",
)
def test_jitted_flow_matches_unjitted_on_trained_flow():
    """jit(compute_flow_actions) must equal its own body with the jit stripped."""
    from envs.env_utils import make_env_and_datasets
    from utils.datasets import Dataset
    from utils.flax_utils import restore_agent

    # Real observations, not Gaussian noise: a trained flow driven from off-distribution
    # states genuinely diverges to NaN, which would mask the effect under test.
    env_name = os.environ.get("PSMFLOWS_FLOW_ENV", "cube-single-play-singletask-v0")
    _, _, train_dataset, _ = make_env_and_datasets(env_name, frame_stack=None)
    batch = Dataset.create(**train_dataset).sample(256)
    obs = jnp.asarray(batch["observations"])

    cfg = get_config()
    cfg["flow_steps"] = 100          # well past the ~30-step onset
    noise = jax.random.normal(jax.random.PRNGKey(0), (obs.shape[0], batch["actions"].shape[-1]))

    agent = FQLAgent.create(0, obs[:1], jnp.asarray(batch["actions"][:1]), cfg)
    agent = restore_agent(agent, os.environ["PSMFLOWS_FLOW_CKPT"],
                          int(os.environ.get("PSMFLOWS_FLOW_CKPT_EPOCH", 500000)))

    jitted = np.asarray(agent.compute_flow_actions(obs, noises=noise))
    unjitted = np.asarray(FQLAgent.compute_flow_actions.__wrapped__(agent, obs, noise))

    err = np.max(np.abs(jitted - unjitted))
    assert err < 1e-3, (
        f"jitted flow integration disagrees with its own body by {err:.4f} on backend "
        f"{jax.default_backend()} — the XLA autotune guard is not in effect "
        f"(XLA_FLAGS={os.environ.get('XLA_FLAGS')!r})."
    )
