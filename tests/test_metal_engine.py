"""
Tests for Parallax Metal / CUDA JIT Engine.
"""

import sys
import numpy as np
import pytest
from parallax.metal_engine import (
    is_metal_available,
    is_cuda_available,
    get_metal_device_name,
    metal_gemm,
    metal_rmsnorm,
)


def test_metal_availability_check():
    avail = is_metal_available()
    assert isinstance(avail, bool)
    if sys.platform == "darwin":
        # On macOS, Metal should be available and device name queryable
        name = get_metal_device_name()
        if avail:
            assert name is not None
            assert len(name) > 0
    else:
        assert not avail
        assert get_metal_device_name() is None


def test_cuda_availability_check():
    cuda_avail = is_cuda_available()
    assert isinstance(cuda_avail, bool)


def test_metal_gemm_correctness():
    np.random.seed(42)
    M, K, N = 16, 32, 24
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    C_metal = metal_gemm(A, B)
    C_expected = A @ B

    assert C_metal.shape == (M, N)
    assert np.allclose(C_metal, C_expected, atol=1e-4)


def test_metal_gemm_invalid_shapes():
    A = np.ones((10, 5), dtype=np.float32)
    B = np.ones((6, 12), dtype=np.float32)
    with pytest.raises(ValueError):
        metal_gemm(A, B)

    with pytest.raises(ValueError):
        metal_gemm(np.ones((10,), dtype=np.float32), B)


def test_metal_rmsnorm_correctness():
    np.random.seed(42)
    rows, cols = 8, 64
    x = np.random.randn(rows, cols).astype(np.float32)
    weight = np.random.uniform(0.5, 1.5, size=(cols,)).astype(np.float32)
    eps = 1e-5

    out_metal = metal_rmsnorm(x, weight, eps=eps)

    variance = np.mean(x**2, axis=-1, keepdims=True)
    expected = (x / np.sqrt(variance + eps)) * weight

    assert out_metal.shape == x.shape
    assert np.allclose(out_metal, expected, atol=1e-5)


def test_metal_rmsnorm_multidimensional():
    np.random.seed(42)
    x = np.random.randn(2, 4, 32).astype(np.float32)
    weight = np.ones(32, dtype=np.float32)

    out = metal_rmsnorm(x, weight)
    assert out.shape == (2, 4, 32)
    variance = np.mean(x**2, axis=-1, keepdims=True)
    expected = (x / np.sqrt(variance + 1e-5)) * weight
    assert np.allclose(out, expected, atol=1e-5)


def test_metal_rmsnorm_invalid_weight():
    x = np.ones((4, 16), dtype=np.float32)
    bad_weight = np.ones((8,), dtype=np.float32)
    with pytest.raises(ValueError):
        metal_rmsnorm(x, bad_weight)
