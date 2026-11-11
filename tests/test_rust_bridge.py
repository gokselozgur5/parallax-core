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
    ver = lib.parallax_version()
    if ver != 2:
        raise RuntimeError(
            f"Rust library ABI version mismatch: expected 2, got {ver}. "
            "Please rebuild with: cargo build --manifest-path crates/parallax-rs/Cargo.toml --release"
        )
    
    # i32 parallax_gemm_f32(m, n, k, alpha, a, a_len, b, b_len, beta, c, c_len)
    lib.parallax_gemm_f32.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.parallax_gemm_f32.restype = ctypes.c_int32

    # i32 parallax_softmax_f32(float* data, size_t data_len, size_t rows, size_t cols)
    lib.parallax_softmax_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.parallax_softmax_f32.restype = ctypes.c_int32

    return lib

def test_rust_bridge_version(rust_lib):
    ver = rust_lib.parallax_version()
    assert ver == 2


def test_rust_bridge_version_mismatch_rejected(monkeypatch):
    class FakeLib:
        def __init__(self):
            self.parallax_version = lambda: 1

    monkeypatch.setattr(ctypes, "CDLL", lambda path: FakeLib())
    with pytest.raises(RuntimeError, match="Rust library ABI version mismatch: expected 2, got 1"):
        # Re-run fixture logic
        lib = ctypes.CDLL("dummy")
        ver = lib.parallax_version()
        if ver != 2:
            raise RuntimeError(
                f"Rust library ABI version mismatch: expected 2, got {ver}. "
                "Please rebuild with: cargo build --manifest-path crates/parallax-rs/Cargo.toml --release"
            )

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

    status = rust_lib.parallax_gemm_f32(
        m, n, k, alpha, a_ptr, a.size, b_ptr, b.size, beta, c_ptr, c.size
    )
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

    status = rust_lib.parallax_gemm_f32(
        m, n, k, alpha, a_ptr, a.size, b_ptr, b.size, beta, c_ptr, c.size
    )
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

    status = rust_lib.parallax_softmax_f32(data_ptr, data_copy.size, rows, cols)
    assert status == 0

    np.testing.assert_allclose(data_copy, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.sum(data_copy, axis=-1), np.ones(rows), rtol=1e-5, atol=1e-5)


def test_rust_bridge_validation_errors(rust_lib):
    buf = np.ones((2, 2), dtype=np.float32)
    ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out = np.zeros((2, 2), dtype=np.float32)
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    assert rust_lib.parallax_gemm_f32(2, 2, 2, 1.0, ptr, 2, ptr, 4, 0.0, out_ptr, 4) == -2
    assert rust_lib.parallax_gemm_f32(2, 2, 2, 1.0, ptr, 4, ptr, 4, 0.0, ptr, 4) == -4
    assert rust_lib.parallax_softmax_f32(ptr, 2, 2, 2) == -2


def test_rust_bridge_pointer_alignment_and_null(rust_lib):
    buf = np.ones((2, 2), dtype=np.float32)
    ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    null_ptr = ctypes.POINTER(ctypes.c_float)()

    # Null pointer checks (-1)
    assert rust_lib.parallax_gemm_f32(2, 2, 2, 1.0, null_ptr, 4, ptr, 4, 0.0, ptr, 4) == -1
    assert rust_lib.parallax_gemm_f32(2, 2, 2, 1.0, ptr, 4, null_ptr, 4, 0.0, ptr, 4) == -1
    assert rust_lib.parallax_gemm_f32(2, 2, 2, 1.0, ptr, 4, ptr, 4, 0.0, null_ptr, 4) == -1
    assert rust_lib.parallax_softmax_f32(null_ptr, 4, 2, 2) == -1

    # Unaligned pointer checks (-3)
    raw = (ctypes.c_char * 64)()
    unaligned = ctypes.cast(ctypes.byref(raw, 1), ctypes.POINTER(ctypes.c_float))
    assert rust_lib.parallax_gemm_f32(2, 2, 2, 1.0, unaligned, 4, ptr, 4, 0.0, ptr, 4) == -3
    assert rust_lib.parallax_gemm_f32(2, 2, 2, 1.0, ptr, 4, unaligned, 4, 0.0, ptr, 4) == -3
    assert rust_lib.parallax_gemm_f32(2, 2, 2, 1.0, ptr, 4, ptr, 4, 0.0, unaligned, 4) == -3
    assert rust_lib.parallax_softmax_f32(unaligned, 4, 2, 2) == -3


def test_rust_bridge_softmax_nan_and_inf(rust_lib):
    data = np.array([[1.0, np.nan, 3.0], [1.0, -np.inf, 2.0]], dtype=np.float32)
    ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    assert rust_lib.parallax_softmax_f32(ptr, data.size, 2, 3) == 0
    assert np.isnan(data[0]).all()
    np.testing.assert_allclose(data[1], [0.26894143, 0.0, 0.7310586], rtol=1e-5, atol=1e-5)


from parallax.rust_engine import (
    find_rust_lib as engine_find_rust_lib,
    is_rust_engine_available,
    rust_gemm,
    rust_softmax,
)


def test_rust_engine_wrapper_availability_and_gemm():
    assert engine_find_rust_lib() is not None
    assert is_rust_engine_available() is True

    m, n, k = 4, 3, 5
    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)
    c = rust_gemm(a, b, alpha=2.0)
    expected = 2.0 * (a @ b)
    np.testing.assert_allclose(c, expected, rtol=1e-5, atol=1e-5)


def test_rust_engine_wrapper_softmax():
    x = np.random.randn(3, 4).astype(np.float32)
    out = rust_softmax(x)
    expected = np.exp(x - np.max(x, axis=-1, keepdims=True))
    expected = expected / np.sum(expected, axis=-1, keepdims=True)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_rust_engine_wrapper_fallback_on_missing(monkeypatch):
    import parallax.rust_engine as re
    monkeypatch.setattr(re, "get_rust_lib", lambda: None)
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(3, 2).astype(np.float32)
    out = re.rust_gemm(a, b)
    np.testing.assert_allclose(out, a @ b, rtol=1e-5, atol=1e-5)

    x = np.array([[-np.inf, -np.inf], [1.0, -np.inf]], dtype=np.float32)
    sm = re.rust_softmax(x)
    assert np.all(sm[0] == 0.0)
    assert np.allclose(sm[1], [1.0, 0.0])
