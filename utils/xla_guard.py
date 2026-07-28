"""Disable XLA:GPU kernel autotuning — it MISCOMPILES the flow integrations.

Import this BEFORE jax, from every entry point that runs on GPU. It only touches
`os.environ`; it must not import jax, because the flag is read when XLA initializes
and setting it afterwards is a silent no-op.

The bug
-------
`FQLAgent.compute_flow_actions` is an unrolled Euler loop over the BC flow. Once the
unroll is long enough, XLA:GPU's autotuner selects a kernel that computes the wrong
thing. Measured on jax 0.10.2 / RTX A5500, cube-single, 256 transitions:

    jitted (any form)   84.5% of outputs pinned at the +-1 clip, mean +0.478
    jit stripped         4.5%,                                   mean +0.042
    manual Euler         4.5%,                                   mean +0.042   (bit-identical)

The jitted result differs from the un-jitted body by 2.0 — the full width of the action
range. It is not drift: the output is garbage saturated at the clip boundaries. The
failure is in the compiled kernel, not the call form (the decorated method, a fresh
`jax.jit`, and the body nested inside an outer jit are all equally wrong), and it is
reproducible across processes. CPU is correct in every form.

It only triggers past a threshold unroll length, which is what bounds the damage:

    flow_steps      5        10       30       100
    max error       5e-4 ok  2e-4 ok  1.21 BAD 2.00 BAD

so training at `flow_steps=10` (configs/agent/fql.yaml) was never affected, while the
100-step diagnostic in tools/validate_flow_inversion.py was reading pure artifact —
it reported round-trip 2.26 for an inverter whose true round-trip is 1.2e-4.

Turning the autotuner off restores agreement to ~5e-5 (`--xla_gpu_deterministic_ops=true`
also works, at more cost). The likely same-root symptom already noted in
tools/validate_flow_inversion.py — un-jitted `vmap` of the inverter returning all-NaN on
GPU — is covered by this too.

Set `PSMFLOWS_ALLOW_XLA_AUTOTUNE=1` to opt out (only to re-measure the bug itself).
"""
import os

_FLAG = "--xla_gpu_autotune_level=0"


def disable_xla_autotune():
    """Append the autotune-off flag to XLA_FLAGS, preserving anything already there."""
    if os.environ.get("PSMFLOWS_ALLOW_XLA_AUTOTUNE"):
        return
    existing = os.environ.get("XLA_FLAGS", "")
    if "xla_gpu_autotune_level" in existing:   # caller already chose a level; respect it
        return
    os.environ["XLA_FLAGS"] = f"{existing} {_FLAG}".strip()


disable_xla_autotune()
