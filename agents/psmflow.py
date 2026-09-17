"""Backward-compat shim. The ``psmflow`` agent was renamed to ``f_psmflow`` on 2026-09-17
(the factorized / affine LatentFlowPSM). It stays importable here, and ``agent=psmflow`` /
``MODE=psmflow`` keep working, so existing checkpoints (whose ``flags.json`` records
``agent_name: psmflow``) and existing scripts are unaffected. New code should import from
``agents.f_psmflow``.
"""
from agents.f_psmflow import *  # noqa: F401,F403
from agents.f_psmflow import _load_flow_params  # noqa: F401  (private; `import *` skips it)
