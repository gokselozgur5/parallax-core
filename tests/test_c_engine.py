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
    np.random.seed(42)
    M, K, N = 64, 128, 32
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    expected = np.matmul(A, B)
    result = c_gemm(A, B)

    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

def test_c_gemm_odd_shapes():
    np.random.seed(1337)
    M, K, N = 7, 13, 19
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    expected = np.matmul(A, B)
    result = c_gemm(A, B)

    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

def test_c_gemm_invalid_shapes():
    A = np.ones((5, 10), dtype=np.float32)
    B = np.ones((8, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        c_gemm(A, B)

    with pytest.raises(ValueError):
        c_gemm(np.ones((2, 2, 2)), np.ones((2, 2)))

def test_c_softmax_correctness():
    np.random.seed(42)
    rows, cols = 32, 64
    X = np.random.randn(rows, cols).astype(np.float32)

    # Reference softmax
    shifted = X - np.max(X, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    expected = exps / np.sum(exps, axis=-1, keepdims=True)

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
