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
