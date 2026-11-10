import numpy as np
import pytest
from unittest.mock import patch

from parallax.c_engine import (
    c_gemm,
    c_softmax,
    is_c_engine_available,
    _find_or_build_library,
)

def test_c_engine_available():
    assert is_c_engine_available() is True
    assert _find_or_build_library() is not None

def test_c_gemm_correctness():
    if not is_c_engine_available():
        pytest.skip("C engine native library not available")
    np.random.seed(42)
    M, K, N = 64, 128, 32
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    expected = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)
    result = c_gemm(A, B)

    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

def test_c_gemm_odd_shapes():
    if not is_c_engine_available():
        pytest.skip("C engine native library not available")
    np.random.seed(1337)
    M, K, N = 7, 13, 19
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    expected = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)
    result = c_gemm(A, B)

    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

def test_c_gemm_zero_inner_dimension():
    # Issue #374: K == 0 must return zeros of shape (M, N)
    A = np.zeros((3, 0), dtype=np.float32)
    B = np.zeros((0, 4), dtype=np.float32)
    result = c_gemm(A, B)
    assert result.shape == (3, 4)
    np.testing.assert_array_equal(result, np.zeros((3, 4), dtype=np.float32))

    # Also test fallback path
    with patch("parallax.c_engine._find_or_build_library", return_value=None):
        result_fallback = c_gemm(A, B)
        assert result_fallback.shape == (3, 4)
        np.testing.assert_array_equal(result_fallback, np.zeros((3, 4), dtype=np.float32))

def test_c_gemm_invalid_shapes():
    A = np.ones((5, 10), dtype=np.float32)
    B = np.ones((8, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        c_gemm(A, B)

    with pytest.raises(ValueError):
        c_gemm(np.ones((2, 2, 2)), np.ones((2, 2)))

def test_c_softmax_correctness():
    if not is_c_engine_available():
        pytest.skip("C engine native library not available")
    np.random.seed(42)
    rows, cols = 32, 64
    X = np.random.randn(rows, cols).astype(np.float32)

    # Reference softmax in float64
    X64 = X.astype(np.float64)
    shifted = X64 - np.max(X64, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    expected = (exps / np.sum(exps, axis=-1, keepdims=True)).astype(np.float32)

    result = c_softmax(X)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.sum(result, axis=-1), np.ones(rows), rtol=1e-5, atol=1e-5)

def test_c_softmax_odd_shapes():
    np.random.seed(101)
    X = np.random.randn(3, 17).astype(np.float32)
    shifted = X - np.max(X, axis=-1, keepdims=True)
    expected = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)

    result = c_softmax(X)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

def test_c_softmax_3d_tensor():
    np.random.seed(7)
    X = np.random.randn(2, 4, 8).astype(np.float32)
    shifted = X - np.max(X, axis=-1, keepdims=True)
    expected = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)

    result = c_softmax(X)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

def test_c_fallback_when_c_engine_unavailable():
    # Test GEMM fallback
    with patch("parallax.c_engine._find_or_build_library", return_value=None):
        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        res = c_gemm(A, B)
        np.testing.assert_allclose(res, A @ B)

        # Test Softmax fallback
        X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        res_sm = c_softmax(X)
        shifted = X - np.max(X, axis=-1, keepdims=True)
        exp_sm = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
        np.testing.assert_allclose(res_sm, exp_sm, rtol=1e-5, atol=1e-5)


<<<<<<< HEAD
def test_concurrent_initialization_c_and_metal():
    import concurrent.futures
    import parallax.c_engine as ce
    import parallax.metal_engine as me

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        c_futures = [executor.submit(ce.is_c_engine_available) for _ in range(8)]
        metal_futures = [executor.submit(me.is_metal_available) for _ in range(8)]

        c_results = [f.result() for f in c_futures]
        metal_results = [f.result() for f in metal_futures]

    assert all(r == c_results[0] for r in c_results)
    assert all(r == metal_results[0] for r in metal_results)


def test_c_engine_diagnostic_reporting_and_require_flag(monkeypatch):
    import os
    import parallax.c_engine as ce

    # Force library load failure
    monkeypatch.setattr(ce, "_C_LIB", None)
    monkeypatch.setattr(ce, "_LIB_LOAD_ATTEMPTED", False)
    monkeypatch.setattr(ce, "_LOAD_ERROR_REASON", None)
    monkeypatch.setattr("ctypes.CDLL", lambda path: (_ for _ in ()).throw(OSError("dummy load error")))

    assert ce.is_c_engine_available() is False
    err = ce.get_c_engine_load_error()
    assert err is not None
    assert "dummy load error" in err

    # Test PARALLAX_REQUIRE_C_ENGINE=1 raises RuntimeError
    monkeypatch.setenv("PARALLAX_REQUIRE_C_ENGINE", "1")
    monkeypatch.setattr(ce, "_C_LIB", None)
    monkeypatch.setattr(ce, "_LIB_LOAD_ATTEMPTED", False)
    with pytest.raises(RuntimeError, match="PARALLAX_REQUIRE_C_ENGINE=1"):
        ce.is_c_engine_available()


@pytest.mark.parametrize("N", [20, 21, 24, 28, 31])
def test_c_gemm_vector_tail_shapes(N):
    if not is_c_engine_available():
        pytest.skip("C engine native library not available")
    M, K = 8, 16
    np.random.seed(N)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    expected = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)
    result = c_gemm(A, B)
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


def test_c_softmax_large_logits_and_special_values():
    if not is_c_engine_available():
        pytest.skip("C engine native library not available")
    X_large = np.array([[100.0, 90.0]], dtype=np.float32)
    res_large = c_softmax(X_large)
    assert not np.isnan(res_large).any()
    np.testing.assert_allclose(res_large[0, 0], 1.0 / (1.0 + np.exp(-10.0)), rtol=1e-5, atol=1e-5)

    X_special = np.array([[1.0, -np.inf, 2.0], [5.0, 5.0, 5.0]], dtype=np.float32)
    res_special = c_softmax(X_special)
    assert res_special[0, 1] == 0.0
    np.testing.assert_allclose(np.sum(res_special[0]), 1.0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_special[1], 1.0 / 3.0, rtol=1e-5, atol=1e-5)


def test_c_engine_unsupported_dtypes():
    with pytest.raises(TypeError):
        c_gemm(np.array([[1 + 1j]]), np.array([[1 + 1j]]))
    with pytest.raises(TypeError):
        c_softmax(np.array([[1 + 1j]]))
    res = c_gemm(np.array([[16777217, 1]], dtype=np.int64), np.array([[1], [0]], dtype=np.int64))
    assert res[0, 0] == 16777217
    res_f64 = c_softmax(np.array([[1e40, 0.0, 1.0]], dtype=np.float64))
    np.testing.assert_allclose(res_f64, [[1.0, 0.0, 0.0]], atol=1e-6)


def test_c_engine_empty_shapes():
    assert c_gemm(np.zeros((0, 4), dtype=np.float32), np.zeros((4, 3), dtype=np.float32)).shape == (0, 3)
    assert c_gemm(np.zeros((2, 4), dtype=np.float32), np.zeros((4, 0), dtype=np.float32)).shape == (2, 0)
    assert c_gemm(np.zeros((2, 0), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)).shape == (2, 3)
    assert c_softmax(np.zeros((0, 4), dtype=np.float32)).shape == (0, 4)
    with pytest.raises(ValueError):
        c_softmax(np.zeros((3, 0), dtype=np.float32))


def test_c_engine_int32_bounds():
    class MockArray:
        ndim = 2
        dtype = np.float32
        shape = (2147483648, 4)
        size = 2147483648 * 4

    b = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="INT32_MAX"):
        c_gemm(MockArray(), b)

    class MockSoftmaxArray:
        ndim = 2
        dtype = np.float32
        shape = (2147483648, 4)
        size = 2147483648 * 4

    with pytest.raises(ValueError, match="INT32_MAX"):
        c_softmax(MockSoftmaxArray())
