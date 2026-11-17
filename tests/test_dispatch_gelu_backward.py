"""Tests for fused GeLU backward functional kernels."""

from __future__ import annotations
import numpy as np
import pytest
from parallax.tensor import Tensor
from python.parallax.dispatch.gelu_backward import (
    COEF,
    SQRT_2_OVER_PI,
    DispatchedGELUBackward,
    dispatched_gelu_backward,
    fused_gelu_backward,
    gelu_backward,
    gelu_backward_reference,
)


def test_constants():
    assert np.isclose(SQRT_2_OVER_PI, 0.7978845608)
    assert np.isclose(COEF, 0.044715)


def test_gelu_backward_reference_parity():
    rng = np.random.default_rng(42)
    x = rng.standard_normal((3, 7)).astype(np.float32)
    dout = rng.standard_normal((3, 7)).astype(np.float32)
    ref = gelu_backward_reference(dout, x, approximate="tanh")
    disp = gelu_backward(dout, x, approximate="tanh")
    np.testing.assert_allclose(ref, disp, rtol=1e-6, atol=1e-6)
    fused = fused_gelu_backward(dout, x, approximate="tanh")
    np.testing.assert_allclose(disp, fused, rtol=1e-6, atol=1e-6)


def test_gelu_backward_finite_difference():
    rng = np.random.default_rng(123)
    x = rng.uniform(-3.0, 3.0, size=(2, 4)).astype(np.float64)
    dout = rng.standard_normal((2, 4)).astype(np.float64)
    analytical = gelu_backward(dout, x, approximate="tanh")
    eps = 1e-4
    numerical = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        xp, xm = x.copy(), x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        up = SQRT_2_OVER_PI * (xp + COEF * (xp**3))
        um = SQRT_2_OVER_PI * (xm + COEF * (xm**3))
        fp = 0.5 * xp * (1.0 + np.tanh(up))
        fm = 0.5 * xm * (1.0 + np.tanh(um))
        numerical[idx] = np.sum(dout * (fp - fm) / (2.0 * eps))
    np.testing.assert_allclose(analytical, numerical, rtol=1e-3, atol=1e-4)
