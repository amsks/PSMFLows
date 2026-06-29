"""Helpers to run a JAX snippet in the isolated OGBench venv from pytest.

The main .venv (which runs pytest) has torch but not jax/flax; jax/flax live
only in /dev/shm/.venv-jax (tmpfs, recreated by `make install-jax`). JAX-side
tests therefore shell out to that interpreter and skip when it is absent.
"""
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SHIM_DIR = REPO / "tools" / "wandb_mode_shim"
OGBENCH_IMPLS = REPO / "third_party" / "ogbench" / "impls"
_JAX_CANDIDATES = (
    "/dev/shm/.venv-jax/bin/python",
    "/mnt/scratch1/.venv-jax/bin/python",
)


def jax_python() -> str:
    for p in _JAX_CANDIDATES:
        if Path(p).exists():
            return p
    pytest.skip("no .venv-jax interpreter found (run `make install-jax`)")


def run_jax(code: str, env_extra: dict | None = None, cwd: str | None = None):
    """Run `code` under the jax interpreter with the shim + OGBench impls on
    PYTHONPATH so `import ogbench_drq` and `import utils.*` both resolve."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SHIM_DIR), str(OGBENCH_IMPLS), env.get("PYTHONPATH", "")]
    )
    env["JAX_PLATFORMS"] = env.get("JAX_PLATFORMS", "cpu")  # tests don't need GPU
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [jax_python(), "-c", textwrap.dedent(code)],
        env=env, cwd=cwd, capture_output=True, text=True,
    )
