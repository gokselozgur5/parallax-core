import ctypes
import os
import platform
import numpy as np
import pytest

def find_rust_lib():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    crate_target = os.path.join(base_dir, "crates", "parallax-rs", "target")
    
    system = platform.system()
    if system == "Darwin":
        lib_name = "libparallax_rs.dylib"
    elif system == "Windows":
        lib_name = "parallax_rs.dll"
    else:
        lib_name = "libparallax_rs.so"

    # Search release first, then debug
    for mode in ["release", "debug"]:
        candidate = os.path.join(crate_target, mode, lib_name)
        if os.path.exists(candidate):
            return candidate
            
    return None

@pytest.fixture(scope="module")
def rust_lib():
    lib_path = find_rust_lib()
    if lib_path is None:
        pytest.skip("Rust library libparallax_rs not compiled yet")
    
    lib = ctypes.CDLL(lib_path)
    
    # Signatures
    # u32 parallax_version()
    lib.parallax_version.argtypes = []
    lib.parallax_version.restype = ctypes.c_uint32
    
    # i32 parallax_gemm_f32(size_t m, size_t n, size_t k, float alpha, const float* a, const float* b, float beta, float* c)
    lib.parallax_gemm_f32.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.parallax_gemm_f32.restype = ctypes.c_int32

    # i32 parallax_softmax_f32(float* data, size_t rows, size_t cols)
    lib.parallax_softmax_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.parallax_softmax_f32.restype = ctypes.c_int32

    return lib

def test_rust_bridge_version(rust_lib):
    ver = rust_lib.parallax_version()
    assert ver == 1

def test_rust_bridge_gemm(rust_lib):
    m, n, k = 4, 3, 5
    np.random.seed(42)
    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)
    c = np.zeros((m, n), dtype=np.float32)

    alpha = 1.0
    beta = 0.0

    a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_ptr = c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    status = rust_lib.parallax_gemm_f32(m, n, k, alpha, a_ptr, b_ptr, beta, c_ptr)
    assert status == 0

    expected = alpha * (a @ b) + beta * c
    np.testing.assert_allclose(c, expected, rtol=1e-5, atol=1e-5)

def test_rust_bridge_gemm_with_beta(rust_lib):
    m, n, k = 2, 2, 3
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    c = np.ones((m, n), dtype=np.float32) * 5.0

    alpha = 2.0
    beta = 0.5

    expected = alpha * (a @ b) + beta * c.copy()

    a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_ptr = c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    status = rust_lib.parallax_gemm_f32(m, n, k, alpha, a_ptr, b_ptr, beta, c_ptr)
    assert status == 0

    np.testing.assert_allclose(c, expected, rtol=1e-5, atol=1e-5)

def test_rust_bridge_softmax(rust_lib):
    rows, cols = 3, 4
    np.random.seed(123)
    data = np.random.randn(rows, cols).astype(np.float32)
    expected = np.exp(data - np.max(data, axis=-1, keepdims=True))
    expected = expected / np.sum(expected, axis=-1, keepdims=True)

    data_copy = data.copy()
    data_ptr = data_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    status = rust_lib.parallax_softmax_f32(data_ptr, rows, cols)
    assert status == 0

    np.testing.assert_allclose(data_copy, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.sum(data_copy, axis=-1), np.ones(rows), rtol=1e-5, atol=1e-5)
