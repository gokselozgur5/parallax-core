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

    special = np.array(
        [[1.0, np.nan, 2.0, 3.0], [np.inf, 1.0, 2.0, 3.0], [-np.inf, 0.0, 1.0, 2.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(softmax_neon(special), c_softmax(special), equal_nan=True)


def _reference_softmax(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    e = np.exp(x64 - x64.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def test_softmax_neon_wide_logit_spread_and_masks() -> None:
    rows = np.array(
        [[0.0, -100.0, 0.0, 0.0], [0.0, -np.inf, 1.0, 2.0], [0.0, -1e9, 1.0, 2.0]],
        dtype=np.float32,
    )
    actual = softmax_neon(rows)
    assert not np.isnan(actual).any()
    np.testing.assert_allclose(actual, _reference_softmax(rows), atol=1e-6)
    np.testing.assert_allclose(actual, c_softmax(rows), atol=1e-6)
    assert actual[1, 1] == 0.0

    mixed = np.array([[2.0, -np.inf, -85.3, -86.5, -150.0, 0.5, -np.inf]], dtype=np.float32)
    np.testing.assert_allclose(softmax_neon(mixed), _reference_softmax(mixed), atol=1e-6)
    np.testing.assert_allclose(softmax_neon(mixed), c_softmax(mixed), atol=1e-6)
