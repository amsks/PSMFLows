"""Production-faithful injection tests.

These drive the OGBench child env through run_gciql.build_child_env (NOT the
test helper's PYTHONPATH) and run from cwd=impls, exactly like a real run. This
matters: sitecustomize is imported at interpreter startup, BEFORE the script
directory is on sys.path, so the shim can only import utils.* if the OGBench
impls dir is on PYTHONPATH. An earlier version of these tests injected impls via
the test helper and so missed that production set PYTHONPATH=shim only.
"""
import os
import subprocess

from run_gciql import build_child_env
from tests.jax_subprocess import OGBENCH_IMPLS, jax_python

_CHECK = """
import utils.encoders as enc
import utils.datasets as ds
print("DRQ", "drq" in enc.encoder_modules)
print("PATCHED", getattr(ds.GCDataset.augment, "_ogbench_drq_patched", False))
"""


def _run_under_production_env(agent_overrides):
    plain = dict(
        env_name="visual-cube-single-play-v0",
        wandb_mode="disabled",
        agent_overrides=agent_overrides,
    )
    env = build_child_env(plain, os.environ.copy())
    env["JAX_PLATFORMS"] = "cpu"
    return subprocess.run(
        [jax_python(), "-c", _CHECK],
        env=env, cwd=str(OGBENCH_IMPLS), capture_output=True, text=True,
    )


def test_drq_frontend_registers_encoder_and_patches_augment():
    proc = _run_under_production_env(
        {"encoder": "drq", "frame_stack": 3, "p_aug": 1.0}
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "DRQ True" in proc.stdout
    assert "PATCHED True" in proc.stdout


def test_impala_frontend_registers_encoder_but_augment_native():
    proc = _run_under_production_env({"encoder": "impala_small", "p_aug": 0.5})
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "DRQ True" in proc.stdout       # registration is unconditional
    assert "PATCHED False" in proc.stdout  # augment swap gated off without flag
