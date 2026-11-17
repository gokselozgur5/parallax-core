"""Tests for fused SwiGLU backward autograd kernel."""

from __future__ import annotations

import numpy as np
import pytest

from python.parallax.dispatch.swiglu_backward import (
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
