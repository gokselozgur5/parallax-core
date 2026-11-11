import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from parallax.c_engine import (
    c_gemm,
    c_rmsnorm,
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


def test_c_gemm_negative_k_returns_immediately():
    # Issue #469: c_gemm_f32 with K < 0 must return without modifying C
    lib = _find_or_build_library()
    if lib is None:
        pytest.skip("C engine native library not available")

    A = np.ones((2, 2), dtype=np.float32)
    B = np.ones((2, 2), dtype=np.float32)
    C = np.full((2, 2), 42.0, dtype=np.float32)

    a_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.c_gemm_f32(a_ptr, b_ptr, c_ptr, 2, 2, -1)
    # C should remain completely untouched (still 42.0)
    np.testing.assert_array_equal(C, np.full((2, 2), 42.0, dtype=np.float32))

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
    # Test GEMM fallback against independent float64 reference
    with patch("parallax.c_engine._find_or_build_library", return_value=None):
        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        res = c_gemm(A, B)
        # Independent reference: float64 loop / hand-calculated values
        expected_gemm = np.zeros((2, 2), dtype=np.float32)
        for i in range(2):
            for j in range(2):
                expected_gemm[i, j] = sum(float(A[i, k]) * float(B[k, j]) for k in range(2))
        np.testing.assert_allclose(res, expected_gemm)

        # Test Softmax fallback against independent reference
        X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        res_sm = c_softmax(X)
        import math
        # Hand-computed reference: max = 3.0, exps = [exp(-2), exp(-1), exp(0)]
        e = [math.exp(float(x) - 3.0) for x in X[0]]
        s = sum(e)
        expected_sm = np.array([[v / s for v in e]], dtype=np.float32)
        np.testing.assert_allclose(res_sm, expected_sm, rtol=1e-5, atol=1e-5)


def test_concurrent_initialization_c_and_metal(monkeypatch):
    import threading
    import concurrent.futures
    import parallax.c_engine as ce
    import parallax.metal_engine as me

    # Reset load state so initialization actually happens under concurrency
    monkeypatch.setattr(ce, "_LIB_LOAD_ATTEMPTED", False)
    monkeypatch.setattr(me, "_METAL_INITIALIZED", False)

    barrier = threading.Barrier(16)

    def init_c():
        barrier.wait()
        return ce.is_c_engine_available()

    def init_metal():
        barrier.wait()
        return me.is_metal_available()

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        c_futures = [executor.submit(init_c) for _ in range(8)]
        metal_futures = [executor.submit(init_metal) for _ in range(8)]

        c_results = [f.result() for f in c_futures]
        metal_results = [f.result() for f in metal_futures]

    assert all(r == c_results[0] for r in c_results)
    assert all(r == metal_results[0] for r in metal_results)


def test_c_engine_diagnostic_reporting_and_require_flag(monkeypatch):
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


def test_c_softmax_nan_and_inf_rows():
    if not is_c_engine_available():
        pytest.skip("C engine native library not available")
    for cols in (3, 16, 17, 32):
        X = np.zeros((3, cols), dtype=np.float32)
        X[0, 0] = 1.0
        X[0, 1] = np.nan
        X[0, 2] = 3.0
        X[1, 0] = 1.0
        X[1, 1] = np.inf
        X[1, 2] = 3.0
        X[2, :] = -np.inf
        res = c_softmax(X)
        assert np.isnan(res).all()

        w = np.ones((cols,), dtype=np.float32)
        x = np.ones((2, cols), dtype=np.float32)
        x[0, cols // 2] = np.nan
        x[1, :] = -np.inf
        res_norm = c_rmsnorm(x, w)
        assert np.isnan(res_norm).all()


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


def test_require_c_engine_raises_consistently_when_unavailable(monkeypatch):
    """Verify PARALLAX_REQUIRE_C_ENGINE=1 raises on repeated calls when C-engine is missing."""
    import parallax.c_engine as ce

    monkeypatch.setenv("PARALLAX_REQUIRE_C_ENGINE", "1")
    monkeypatch.setattr(ce, "_C_LIB", None)
    monkeypatch.setattr(ce, "_LIB_LOAD_ATTEMPTED", True)
    monkeypatch.setattr(ce, "_LOAD_ERROR_REASON", "Library not found")

    with pytest.raises(RuntimeError, match="PARALLAX_REQUIRE_C_ENGINE=1"):
        ce._find_or_build_library()

    with pytest.raises(RuntimeError, match="PARALLAX_REQUIRE_C_ENGINE=1"):
        ce._find_or_build_library()


def test_makefile_builds_with_default_cc():
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_dir = Path(tmpdir) / "bin"
        bin_dir.mkdir()
        # Mock a failing clang in PATH to ensure Makefile uses cc and does not force clang
        clang_mock = bin_dir / "clang"
        clang_mock.write_text("#!/bin/sh\necho 'clang: No such file or directory' >&2\nexit 127\n")
        clang_mock.chmod(0o755)

        src_dir = Path(tmpdir) / "csrc"
        csrc_repo = Path(__file__).parent.parent / "parallax" / "csrc"
        shutil.copytree(csrc_repo, src_dir)

        env = dict(os.environ)
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
        res = subprocess.run(["make", "-C", str(src_dir)], env=env, capture_output=True, text=True)
        assert res.returncode == 0, f"Makefile build failed with cc: {res.stderr}"


def test_rebuild_on_source_change_without_stale_cache():
    with tempfile.TemporaryDirectory() as td:
        fake_repo = Path(td) / "repo"
        repo_root = Path(__file__).parent.parent
        csrc_src = repo_root / "parallax" / "csrc"
        c_engine_src = repo_root / "parallax" / "c_engine.py"

        (fake_repo / "parallax" / "csrc").mkdir(parents=True)
        (fake_repo / "parallax" / "__init__.py").write_text("")
        shutil.copy2(c_engine_src, fake_repo / "parallax" / "c_engine.py")
        for f in csrc_src.glob("*"):
            if f.is_file():
                shutil.copy2(f, fake_repo / "parallax" / "csrc" / f.name)

        cache_base = Path(td) / "cache"
        env = {**os.environ, "XDG_CACHE_HOME": str(cache_base), "PYTHONPATH": str(fake_repo)}

        cmd1 = "import parallax.c_engine as ce; assert ce.is_c_engine_available()"
        res1 = subprocess.run([sys.executable, "-c", cmd1], env=env, cwd=str(fake_repo), capture_output=True, text=True)
        assert res1.returncode == 0, f"Initial build failed: {res1.stderr}"

        # Corrupt a source file
        softmax_c = fake_repo / "parallax" / "csrc" / "softmax.c"
        softmax_c.write_text("SYNTAX ERROR IN C FILE\n" + softmax_c.read_text())

        cmd2 = "import parallax.c_engine as ce; assert not ce.is_c_engine_available()"
        res2 = subprocess.run([sys.executable, "-c", cmd2], env=env, cwd=str(fake_repo), capture_output=True, text=True)
        assert res2.returncode == 0, f"Expected broken build to fail but passed: {res2.stderr}"
def test_c_engine_build_timeout_and_cache_directory(monkeypatch):
    """Verify build into user cache directory and timeout error handling (closes #437)."""
    import subprocess
    import tempfile
    from pathlib import Path
    import parallax.c_engine as ce

    with tempfile.TemporaryDirectory() as tmp_cache:
        monkeypatch.setattr(ce, "_C_LIB", None)
        monkeypatch.setattr(ce, "_LIB_LOAD_ATTEMPTED", False)
        monkeypatch.setattr(ce, "_LOAD_ERROR_REASON", None)
        monkeypatch.setattr(ce, "_get_build_cache_dir", lambda: Path(tmp_cache))

        orig_run = subprocess.run

        def mock_run(cmd, *args, **kwargs):
            if "make" in cmd:
                timeout = kwargs.get("timeout")
                assert timeout is not None, "make called without timeout"
                assert timeout == 15
                raise subprocess.TimeoutExpired(cmd, timeout)
            return orig_run(cmd, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", mock_run)

        lib = ce._find_or_build_library_unlocked()
        assert lib is None
        reason = ce.get_c_engine_load_error()
        assert reason is not None
        assert "timed out" in reason.lower()


def test_c_rmsnorm_fallback_narrow_integer_precision():
    """Verify c_rmsnorm fallback does not overflow on narrow integer types (issue #474)."""
    x_int = np.array([[100, 200], [-100, -200]], dtype=np.int16)
    weight = np.array([1.0, 1.0], dtype=np.float32)
    with patch("parallax.c_engine._find_or_build_library", return_value=None):
        out = c_rmsnorm(x_int, weight)
        ref_x = x_int.astype(np.float32)
        ref_rms = np.sqrt(np.mean(ref_x ** 2, axis=-1, keepdims=True) + 1e-6)
        expected = (ref_x / ref_rms) * weight
        assert not np.isnan(out).any()
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
