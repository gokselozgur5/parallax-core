import numpy as np
import pytest
from unittest.mock import patch

from parallax.c_engine import (
    c_rmsnorm,
    is_c_engine_available,
    _find_or_build_library,
)
from parallax.unified_norm import UnifiedNorm


def test_c_rmsnorm_available():
    assert is_c_engine_available() is True
    assert _find_or_build_library() is not None


def test_c_rmsnorm_numerical_parity_f32():
    if not is_c_engine_available():
        pytest.skip("C engine native library not available")
    np.random.seed(42)
    rows, cols = 32, 128
    x = np.random.randn(rows, cols).astype(np.float32)
    weight = np.random.randn(cols).astype(np.float32)
    eps = 1e-6

    x64 = x.astype(np.float64)
    w64 = weight.astype(np.float64)
    rms = np.sqrt(np.mean(x64 ** 2, axis=-1, keepdims=True) + float(eps))
    expected = ((x64 / rms) * w64).astype(np.float32)

    actual = c_rmsnorm(x, weight, eps=eps)

    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_c_rmsnorm_numerical_parity_f64():
    if not is_c_engine_available():
        pytest.skip("C engine native library not available")
    np.random.seed(42)
    rows, cols = 16, 64
    x = np.random.randn(rows, cols).astype(np.float64)
    weight = np.random.randn(cols).astype(np.float64)
    eps = 1e-6

    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    expected = (x / rms) * weight

    actual = c_rmsnorm(x, weight, eps=eps)

    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("cols", [63, 65, 37, 7, 1])
def test_c_rmsnorm_numerical_parity_f64_odd_columns(cols):
    if not is_c_engine_available():
        pytest.skip("C engine native library not available")
    np.random.seed(cols)
    rows = 4
    x = np.random.randn(rows, cols).astype(np.float64)
    weight = np.random.randn(cols).astype(np.float64)
    eps = 1e-6
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    expected = (x / rms) * weight
    actual = c_rmsnorm(x, weight, eps=eps)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_c_rmsnorm_odd_shapes():
    np.random.seed(1337)
    rows, cols = 7, 37
    x = np.random.randn(rows, cols).astype(np.float32)
    weight = np.random.randn(cols).astype(np.float32)
    eps = 1e-5

    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    expected = (x / rms) * weight

    actual = c_rmsnorm(x, weight, eps=eps)

    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_c_rmsnorm_multidimensional_tensor():
    np.random.seed(777)
    shape = (2, 4, 16)
    x = np.random.randn(*shape).astype(np.float32)
    weight = np.random.randn(16).astype(np.float32)
    eps = 1e-6

    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    expected = (x / rms) * weight

    actual = c_rmsnorm(x, weight, eps=eps)

    assert actual.shape == shape
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_c_rmsnorm_invalid_weight_shape():
    x = np.ones((4, 8), dtype=np.float32)
    weight = np.ones((10,), dtype=np.float32)
    with pytest.raises(ValueError):
        c_rmsnorm(x, weight)


def test_c_rmsnorm_fallback_when_unavailable():
    np.random.seed(123)
    x = np.random.randn(4, 16).astype(np.float32)
    weight = np.random.randn(16).astype(np.float32)
    eps = 1e-6

    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    expected = (x / rms) * weight

    with patch("parallax.c_engine._find_or_build_library", return_value=None):
        actual = c_rmsnorm(x, weight, eps=eps)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_unified_norm_uses_c_rmsnorm_and_matches_numpy():
    np.random.seed(42)
    dim = 64
    x = np.random.randn(3, 10, dim).astype(np.float32)
    residual = np.random.randn(3, 10, dim).astype(np.float32)

    norm = UnifiedNorm(dim=dim, eps=1e-6, norm_type="rmsnorm")
    norm.weight = np.random.randn(dim).astype(np.float32)

    # 1. Forward with C engine
    out_c, fused_c = norm.forward(x, residual=residual)

    # 2. Forward fallback (mock C engine unavailable)
    with patch("parallax.unified_norm.is_c_engine_available", return_value=False):
        out_py, fused_py = norm.forward(x, residual=residual)

    np.testing.assert_allclose(fused_c, fused_py, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(out_c, out_py, atol=1e-6, rtol=1e-5)
