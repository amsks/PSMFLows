"""Reusable weight-transplant + numerical-match helpers for parity gates.

These helpers are the scaffolding sub-projects 2 (RLDP + RLDP+FlowBC on state)
and 3 (pixel variants) use to verify that our torch port of td_jepa's RLDP
produces numerically equivalent outputs to the reference implementation.

Usage pattern (in a sub-project 2/3 test file):

    src = load_jax_checkpoint("/path/to/td_jepa/checkpoint.safetensors")
    torch_model = OurRLDPModel(...)
    mapping = {"backward_map.weight": "params/backward_map/kernel", ...}
    transplant(src, torch_model, mapping)
    assert_forward_match(jax_forward, torch_model, inputs, atol=1e-5, rtol=1e-5)
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import torch
from torch import nn


def load_jax_checkpoint(path: str) -> Dict[str, np.ndarray]:
    """Load a safetensors file (the canonical td_jepa save format) into a
    leaf-name -> numpy array dict. td_jepa's FBModel.save() writes
    `model.safetensors`."""
    import safetensors.numpy
    loaded = safetensors.numpy.load_file(path)
    return dict(loaded)


def transplant(
    src_params: Dict[str, np.ndarray],
    torch_module: nn.Module,
    mapping: Dict[str, str],
) -> None:
    """Copy values from `src_params` into `torch_module.state_dict()`.

    `mapping`: dict of torch_param_name -> src_param_name. The src is assumed
    to follow JAX/flax conventions where Linear layers store kernels as
    (in_dim, out_dim); torch stores (out_dim, in_dim). Kernels are transposed
    when the mapped torch name ends with '.weight' AND the rank is 2.
    Bias and higher-rank tensors are copied as-is (callers can override by
    pre-transposing the src values if needed for non-standard layouts).

    Raises:
        KeyError: if any torch parameter is not in the mapping, or any mapped
                  src name is missing from `src_params`.
    """
    torch_keys = set(dict(torch_module.named_parameters()).keys())
    mapped_torch_keys = set(mapping.keys())
    unassigned = torch_keys - mapped_torch_keys
    if unassigned:
        raise KeyError(
            f"transplant: torch params not in mapping: {sorted(unassigned)}"
        )
    new_state = dict(torch_module.state_dict())
    for tname, sname in mapping.items():
        if sname not in src_params:
            raise KeyError(f"transplant: src param '{sname}' not found")
        val = torch.as_tensor(src_params[sname])
        if tname.split(".")[-1] == "weight" and val.ndim == 2:
            val = val.T.contiguous()
        new_state[tname] = val
    torch_module.load_state_dict(new_state, strict=True)


def assert_forward_match(
    ref_fn: Callable,
    ours_fn: Callable,
    inputs,
    atol: float,
    rtol: float,
    label: str = "",
) -> None:
    """Call both functions on `inputs`, assert torch.allclose.

    `inputs` may be a single tensor or a tuple of args. On mismatch, raises
    AssertionError including the max abs diff between the two outputs."""
    args = inputs if isinstance(inputs, tuple) else (inputs,)
    ref_out = ref_fn(*args)
    ours_out = ours_fn(*args)
    ref_t = torch.as_tensor(ref_out)
    ours_t = torch.as_tensor(ours_out)
    if not torch.allclose(ref_t, ours_t, atol=atol, rtol=rtol):
        max_diff = (ref_t - ours_t).abs().max().item()
        raise AssertionError(
            f"[parity:{label}] forward mismatch (atol={atol}, rtol={rtol}): "
            f"max abs diff = {max_diff:.3e}; ref shape={tuple(ref_t.shape)}, "
            f"ours shape={tuple(ours_t.shape)}"
        )


def assert_grad_match(
    ref_module: nn.Module,
    ours_module: nn.Module,
    loss_fn: Callable[[nn.Module, "torch.Tensor"], "torch.Tensor"],
    inputs,
    atol: float,
    rtol: float,
    lr: float = 1e-3,
) -> None:
    """Run one SGD step on each module with the same `loss_fn(module, inputs)`,
    then assert post-step state_dict parity. Modules are assumed to start with
    identical weights (caller's responsibility — typically via transplant()).

    Raises AssertionError on the first parameter that diverges beyond tolerance.
    """
    opt_ref = torch.optim.SGD(ref_module.parameters(), lr=lr)
    opt_ours = torch.optim.SGD(ours_module.parameters(), lr=lr)
    opt_ref.zero_grad()
    opt_ours.zero_grad()
    loss_fn(ref_module, inputs).backward()
    loss_fn(ours_module, inputs).backward()
    opt_ref.step()
    opt_ours.step()
    ref_state = ref_module.state_dict()
    ours_state = ours_module.state_dict()
    for name, ref_v in ref_state.items():
        ours_v = ours_state[name]
        if not torch.allclose(ref_v, ours_v, atol=atol, rtol=rtol):
            diff = (ref_v - ours_v).abs().max().item()
            raise AssertionError(
                f"[parity:grad] param '{name}' diverged after 1 step: "
                f"max abs diff = {diff:.3e} (atol={atol}, rtol={rtol})"
            )
