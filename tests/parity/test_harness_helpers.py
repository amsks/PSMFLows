"""Unit tests for the parity-gate harness helpers.

The harness is the shared scaffolding sub-projects 2 and 3 use to weight-transplant
td_jepa-trained RLDP modules into our torch implementations and assert numerical
equivalence. These tests exercise the helpers on toy synthetic modules so we know
the harness itself is correct before relying on it for real parity gates.
"""
import numpy as np
import pytest
import torch
from torch import nn

from tests.parity.harness import (
    transplant,
    assert_forward_match,
    assert_grad_match,
)


def test_transplant_copies_linear_weights_with_jax_shape():
    """JAX linear stores W as (in, out); torch as (out, in). transplant must
    transpose. Bias has the same shape in both."""
    torch_lin = nn.Linear(3, 4)
    src = {
        "linear.kernel": np.arange(12, dtype=np.float32).reshape(3, 4),
        "linear.bias": np.arange(4, dtype=np.float32),
    }
    mapping = {"weight": "linear.kernel", "bias": "linear.bias"}
    transplant(src, torch_lin, mapping)
    assert torch.allclose(torch_lin.weight, torch.as_tensor(src["linear.kernel"].T))
    assert torch.allclose(torch_lin.bias, torch.as_tensor(src["linear.bias"]))


def test_transplant_raises_on_unassigned_torch_param():
    """If any torch param is missing from the mapping, raise so the user notices."""
    torch_lin = nn.Linear(3, 4)
    src = {"linear.kernel": np.zeros((3, 4), dtype=np.float32)}
    mapping = {"weight": "linear.kernel"}  # 'bias' omitted on purpose
    with pytest.raises(KeyError, match="bias"):
        transplant(src, torch_lin, mapping)


def test_transplant_raises_on_unknown_src_key():
    """If a mapping points at a missing src key, raise immediately."""
    torch_lin = nn.Linear(3, 4)
    src = {"linear.kernel": np.zeros((3, 4), dtype=np.float32)}
    mapping = {"weight": "linear.kernel", "bias": "linear.bias"}
    with pytest.raises(KeyError, match="linear.bias"):
        transplant(src, torch_lin, mapping)


def test_assert_forward_match_passes_for_identical_fns():
    """Two identical functions on the same input must satisfy the gate."""
    def f1(x): return x * 2.0
    def f2(x): return x * 2.0
    x = torch.randn(4)
    assert_forward_match(f1, f2, x, atol=1e-7, rtol=1e-7, label="trivial")


def test_assert_forward_match_raises_on_mismatch():
    """When outputs diverge beyond tolerance, raise AssertionError with diff info."""
    def f1(x): return x * 2.0
    def f2(x): return x * 2.1
    x = torch.randn(4)
    with pytest.raises(AssertionError, match="max abs diff"):
        assert_forward_match(f1, f2, x, atol=1e-5, rtol=1e-5, label="mismatch")


def test_assert_grad_match_after_one_step_on_identical_modules():
    """Two clones starting from the same params and stepped on the same loss
    must remain identical."""
    torch.manual_seed(0)
    m_ref = nn.Linear(3, 4)
    m_ours = nn.Linear(3, 4)
    m_ours.load_state_dict(m_ref.state_dict())
    x = torch.randn(2, 3)

    def loss_fn(module, inputs):
        return module(inputs).pow(2).sum()

    assert_grad_match(m_ref, m_ours, loss_fn, x, atol=1e-7, rtol=1e-7)


def test_assert_grad_match_raises_on_diverged_modules():
    """If the modules diverge after one step, raise."""
    torch.manual_seed(0)
    m_ref = nn.Linear(3, 4)
    m_ours = nn.Linear(3, 4)
    x = torch.randn(2, 3)

    def loss_fn(module, inputs):
        return module(inputs).pow(2).sum()

    with pytest.raises(AssertionError):
        assert_grad_match(m_ref, m_ours, loss_fn, x, atol=1e-7, rtol=1e-7)
