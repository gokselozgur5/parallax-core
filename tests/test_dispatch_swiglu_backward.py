"""Tests for fused SwiGLU backward autograd kernel."""

from __future__ import annotations

import numpy as np
import pytest

from parallax.tensor import Tensor
from parallax.dispatch.swiglu_backward import (
    dispatched_swiglu_backward,
    gated_mlp_backward,
    swiglu_backward,
    swiglu_backward_reference,
)


def test_swiglu_backward_reference_numerical() -> None:
    np.random.seed(42)
    g = np.random.randn(2, 4).astype(np.float64)
    u = np.random.randn(2, 4).astype(np.float64)
    dout = np.random.randn(2, 4).astype(np.float64)
    dg, du = swiglu_backward_reference(dout, g, u)
    h = 1e-6
    f = lambda a, b: a * (1.0 / (1.0 + np.exp(-a))) * b
    num_dg = (f(g + h, u) - f(g - h, u)) / (2 * h) * dout
    num_du = (f(g, u + h) - f(g, u - h)) / (2 * h) * dout
    np.testing.assert_allclose(dg, num_dg, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(du, num_du, rtol=1e-5, atol=1e-5)


def test_swiglu_backward_validation() -> None:
    g = np.ones((2, 3), dtype=np.float32)
    u = np.ones((2, 4), dtype=np.float32)
    dout = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        swiglu_backward_reference(dout, g, u)
    with pytest.raises(ValueError, match="Grad shape mismatch"):
        swiglu_backward_reference(np.ones((2, 4), dtype=np.float32), g, g)
    with pytest.raises(TypeError, match="Complex"):
        swiglu_backward_reference(dout, g + 1j, g)


def test_gated_mlp_backward_shapes_and_values() -> None:
    np.random.seed(42)
    x = np.random.randn(2, 4, 8).astype(np.float32)
    w_gate = np.random.randn(16, 8).astype(np.float32)
    w_up = np.random.randn(16, 8).astype(np.float32)
    w_down = np.random.randn(8, 16).astype(np.float32)
    gate = x @ w_gate.T
    up = x @ w_up.T
    dout = np.ones_like(x)
    dx, dw_gate, dw_up, dw_down = gated_mlp_backward(
        dout, x, gate, up, w_gate, w_up, w_down
    )
    assert dx.shape == x.shape
    assert dw_gate.shape == w_gate.shape
    assert dw_up.shape == w_up.shape
    assert dw_down.shape == w_down.shape


def test_dispatched_swiglu_backward_autograd() -> None:
    g = Tensor(np.array([[1.0, -2.0], [0.5, 1.5]], dtype=np.float32), requires_grad=True)
    u = Tensor(np.array([[2.0, 3.0], [-1.0, 0.5]], dtype=np.float32), requires_grad=True)
    dout = Tensor(np.ones((2, 2), dtype=np.float32))
    dg, du = dispatched_swiglu_backward(dout, g, u)
    assert g.grad is not None
    assert u.grad is not None
    assert g.grad.shape == g.shape
    assert u.grad.shape == u.shape


def test_fused_swiglu_backward_exports() -> None:
    import parallax.dispatch as pd

    assert hasattr(pd, "FusedSwiGLUBackward")
    assert hasattr(pd, "dispatched_swiglu_backward")
    assert hasattr(pd, "gated_mlp_backward")
    assert hasattr(pd, "swiglu_backward")
    assert hasattr(pd, "swiglu_backward_reference")
    fb = pd.FusedSwiGLUBackward
    assert callable(fb.backward)
    assert callable(fb.mlp_backward)
    assert callable(fb.reference)


def test_swiglu_backward_parity() -> None:
    g = np.array([1.0, -1.0], dtype=np.float32)
    u = np.array([2.0, 3.0], dtype=np.float32)
    dout = np.array([0.5, -0.5], dtype=np.float32)
    dg, du = swiglu_backward(dout, g, u)
    dg_ref, du_ref = swiglu_backward_reference(dout, g, u)
    np.testing.assert_allclose(dg, dg_ref)
    np.testing.assert_allclose(du, du_ref)
