"""Tests for fused SwiGLU backward autograd kernel."""

from __future__ import annotations

import numpy as np
import pytest

from python.parallax.dispatch.swiglu_backward import (
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
