"""Unit tests for SwiGLU activation layer and dispatch helpers."""

import numpy as np
import pytest
from python.parallax.dispatch.swiglu import swiglu_dispatch


def test_swiglu_dispatch_parity() -> None:
    a = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = swiglu_dispatch(a, b)
    sig = 1.0 / (1.0 + np.exp(-a))
    expected = a * sig * b
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
