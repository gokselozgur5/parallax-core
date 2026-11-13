"""Tests for NEON polynomial exp softmax microkernel."""

import platform
import numpy as np
import pytest

if platform.machine() not in ("arm64", "aarch64"):
    pytest.skip("NEON softmax requires arm64", allow_module_level=True)

from parallax.c_engine import c_softmax
from parallax.native.softmax_neon import is_softmax_neon_available, softmax_neon


def test_softmax_neon_relative_error_and_nan_inf():
    assert is_softmax_neon_available()
    rng = np.random.default_rng(123)
    x = rng.standard_normal((4, 33)).astype(np.float32)
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    expected = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    actual = softmax_neon(x)
    rel_err = np.abs(actual - expected) / np.maximum(expected, 1e-12)
    assert np.max(rel_err) < 1e-5

    special = np.array([[1.0, np.nan, 2.0], [np.inf, 1.0, 2.0], [-np.inf, 0.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(softmax_neon(special), c_softmax(special), equal_nan=True)
