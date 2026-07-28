"""Pytest bootstrap.

The ONLY job here is to run utils.xla_guard before anything imports jax. pytest
imports test modules in filename order, and most of them pull in `agents.*` (hence
jax) at module scope, so a guard imported inside a test function is already too late —
XLA reads XLA_FLAGS when it initialises, and setting it afterwards is a silent no-op.

Without this, GPU test runs use the autotuned kernels that miscompile the flow
integration; see utils/xla_guard.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401,E402  -- must precede any jax import
