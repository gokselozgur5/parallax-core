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
        assert avail is True, "Metal must be available and initialized on macOS / Darwin"
        name = get_metal_device_name()
        assert name is not None
        assert len(name) > 0
    else:
        assert not avail
        assert get_metal_device_name() is None


def test_metal_gpu_path_active_and_no_silent_fallback():
    if sys.platform != "darwin":
        pytest.skip("Metal only available on Darwin")
    assert is_metal_available() is True, "Metal GPU engine failed to initialize"
    import parallax.metal_engine as me
    assert me._PSO_GEMM is not None, "Metal GEMM pipeline state object must be initialized"
    assert me._PSO_RMSNORM is not None, "Metal RMSNorm pipeline state object must be initialized"


def test_cuda_availability_check():
    cuda_avail = is_cuda_available()
    assert isinstance(cuda_avail, bool)


def test_metal_gemm_correctness():
    np.random.seed(42)
    M, K, N = 16, 32, 24
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    C_metal = metal_gemm(A, B)
    C_expected = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)

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

    x64 = x.astype(np.float64)
    w64 = weight.astype(np.float64)
    var64 = np.mean(x64**2, axis=-1, keepdims=True)
    expected = ((x64 / np.sqrt(var64 + float(eps))) * w64).astype(np.float32)

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


<<<<<<< HEAD
def test_metal_buffers_released_no_memory_leak():
    if not is_metal_available() or sys.platform != "darwin":
        pytest.skip("Metal not available on this platform")

    import ctypes
    from parallax.metal_engine import _OBJC, _DEVICE

    objc_msgSend = ctypes.cast(_OBJC.objc_msgSend, ctypes.c_void_p).value
    sel = _OBJC.sel_registerName(b"currentAllocatedSize")
    query_alloc = ctypes.CFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p)(objc_msgSend)

    # Warmup
    A = np.random.randn(128, 128).astype(np.float32)
    B = np.random.randn(128, 128).astype(np.float32)
    metal_gemm(A, B)

    alloc_before = query_alloc(_DEVICE, sel)
    for _ in range(50):
        metal_gemm(A, B)
    alloc_after = query_alloc(_DEVICE, sel)
    assert alloc_after == alloc_before, f"GEMM leaked memory: {alloc_after - alloc_before} bytes"

    # Test RMSNorm
    x = np.random.randn(64, 128).astype(np.float32)
    w = np.random.randn(128).astype(np.float32)
    metal_rmsnorm(x, w)

    alloc_before_rms = query_alloc(_DEVICE, sel)
    for _ in range(50):
        metal_rmsnorm(x, w)
    alloc_after_rms = query_alloc(_DEVICE, sel)
    assert alloc_after_rms == alloc_before_rms, f"RMSNorm leaked memory: {alloc_after_rms - alloc_before_rms} bytes"


def test_metal_engine_types_and_empty_shapes():
    with pytest.raises(TypeError):
        metal_gemm(np.ones((2, 2), dtype=np.complex64), np.ones((2, 2), dtype=np.complex64))
    with pytest.raises(TypeError):
        metal_rmsnorm(np.ones((2, 4), dtype=np.complex64), np.ones(4, dtype=np.float32))

    assert metal_gemm(np.zeros((0, 4), dtype=np.float32), np.zeros((4, 3), dtype=np.float32)).shape == (0, 3)
    assert metal_gemm(np.zeros((2, 4), dtype=np.float32), np.zeros((4, 0), dtype=np.float32)).shape == (2, 0)
    assert metal_rmsnorm(np.zeros((0, 8), dtype=np.float32), np.ones(8, dtype=np.float32)).shape == (0, 8)
    with pytest.raises(ValueError):
        metal_rmsnorm(np.zeros((3, 0), dtype=np.float32), np.ones(0, dtype=np.float32))

    f64_res = metal_rmsnorm(np.array([[1e150]], dtype=np.float64), np.ones(1, dtype=np.float64))
    assert not np.isnan(f64_res).any()
