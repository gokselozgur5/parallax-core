"""
Parallax C-Engine: High-performance C implementations for GEMM and Softmax with ctypes and NumPy fallback.
"""

from __future__ import annotations

import ctypes
import os
import sys
import shutil
import tempfile
import subprocess
import threading
import logging
from pathlib import Path
from typing import Optional
import numpy as np

_C_LIB: Optional[ctypes.CDLL] = None
_LIB_LOAD_ATTEMPTED: bool = False
_INIT_LOCK = threading.Lock()
_LOAD_ERROR_REASON: Optional[str] = None
_LOGGER = logging.getLogger("parallax.c_engine")
_INT32_MAX: int = 2147483647


def _get_build_cache_dir() -> Path:
    cache_base = os.environ.get("XDG_CACHE_HOME")
    if cache_base:
        p = Path(cache_base) / "parallax" / "c_engine"
    else:
        p = Path.home() / ".cache" / "parallax" / "c_engine"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _find_candidate_in_dir(d: Path, candidate_names: list[str]) -> Optional[Path]:
    for name in candidate_names:
        candidate = d / name
        if candidate.is_file():
            return candidate
    return None

def _find_or_build_library() -> Optional[ctypes.CDLL]:
    global _C_LIB, _LIB_LOAD_ATTEMPTED
    if _LIB_LOAD_ATTEMPTED:
        return _C_LIB

    with _INIT_LOCK:
        if _LIB_LOAD_ATTEMPTED:
            return _C_LIB
        return _find_or_build_library_unlocked()


def _find_or_build_library_unlocked() -> Optional[ctypes.CDLL]:
    global _C_LIB, _LIB_LOAD_ATTEMPTED, _LOAD_ERROR_REASON
    csrc_dir = Path(__file__).parent / "csrc"
    
    # Candidate shared library names
    exts = [".dylib", ".so", ".dll"]
    candidate_names = [f"libparallax_c{ext}" for ext in exts]
    
    lib_path = _find_candidate_in_dir(csrc_dir, candidate_names)
    cache_dir = None
    if lib_path is None:
        try:
            cache_dir = _get_build_cache_dir()
            lib_path = _find_candidate_in_dir(cache_dir, candidate_names)
        except Exception:
            pass

    # If not found, attempt compilation
    if lib_path is None and (csrc_dir / "Makefile").is_file():
        try:
            if cache_dir is None:
                cache_dir = _get_build_cache_dir()
            for src_file in csrc_dir.glob("*"):
                if src_file.is_file():
                    shutil.copy2(src_file, cache_dir / src_file.name)
            subprocess.run(["make", "-C", str(cache_dir)], check=True, capture_output=True, timeout=15)
            lib_path = _find_candidate_in_dir(cache_dir, candidate_names)
        except subprocess.TimeoutExpired as e:
            _LOAD_ERROR_REASON = f"Build timed out after 15 seconds: {e}"
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode("utf-8", errors="replace").strip() if e.stderr else str(e)
            _LOAD_ERROR_REASON = f"Compilation failed: {err_msg}"
        except Exception as e:
            _LOAD_ERROR_REASON = f"Build error: {e}"

    if lib_path is None and _LOAD_ERROR_REASON is None:
        _LOAD_ERROR_REASON = "Shared library not found and no compilation candidate succeeded"

    if lib_path is not None and lib_path.is_file():
        try:
            lib = ctypes.CDLL(str(lib_path))
            # void c_gemm_f32(const float *A, const float *B, float *C, int M, int N, int K);
            lib.c_gemm_f32.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.c_gemm_f32.restype = None

            # void c_softmax_f32(const float *input, float *output, int rows, int cols);
            lib.c_softmax_f32.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.c_softmax_f32.restype = None

            # void c_rmsnorm_f32(const float *x, const float *weight, float *output, int rows, int cols, float eps);
            lib.c_rmsnorm_f32.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
            ]
            lib.c_rmsnorm_f32.restype = None

            # void c_rmsnorm_f64(const double *x, const double *weight, double *output, int rows, int cols, double eps);
            lib.c_rmsnorm_f64.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_double,
            ]
            lib.c_rmsnorm_f64.restype = None
            _C_LIB = lib
            _LOAD_ERROR_REASON = None
        except Exception as e:
            _C_LIB = None
            _LOAD_ERROR_REASON = f"Failed to load shared library {lib_path}: {e}"

    _LIB_LOAD_ATTEMPTED = True
    if _C_LIB is None:
        if os.environ.get("PARALLAX_REQUIRE_C_ENGINE") == "1":
            raise RuntimeError(f"PARALLAX_REQUIRE_C_ENGINE=1 but C-engine failed: {_LOAD_ERROR_REASON}")
        _LOGGER.warning("C-engine unavailable, falling back to NumPy. Reason: %s", _LOAD_ERROR_REASON)

    return _C_LIB

def is_c_engine_available() -> bool:
    """Return True if the native C-engine library is compiled and loaded."""
    return _find_or_build_library() is not None

def get_c_engine_load_error() -> Optional[str]:
    """Return the error message explaining why C-engine failed to load, or None if loaded."""
    _find_or_build_library()
    return _LOAD_ERROR_REASON

def c_gemm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute matrix multiplication C = A @ B using C-engine if available,
    otherwise fallback to NumPy.
    A: 2D array of shape (M, K), float32
    B: 2D array of shape (K, N), float32
    """
    if np.iscomplexobj(A) or np.iscomplexobj(B):
        raise TypeError("Complex dtypes are not supported")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"Inputs must be 2D matrices, got shapes {A.shape} and {B.shape}")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible shapes for GEMM: {A.shape} and {B.shape}")

    M, K = A.shape
    K2, N = B.shape
    if M > _INT32_MAX or N > _INT32_MAX or K > _INT32_MAX or (M * K) > _INT32_MAX or (K * N) > _INT32_MAX or (M * N) > _INT32_MAX:
        raise ValueError(f"Matrix dimensions exceed INT32_MAX: ({M}, {K}) @ ({K2}, {N})")

    if M == 0 or N == 0 or K == 0 or A.dtype != np.float32 or B.dtype != np.float32:
        return np.matmul(A, B)

    lib = _find_or_build_library()
    if lib is None:
        return np.matmul(A, B)

    A_c = np.ascontiguousarray(A, dtype=np.float32)
    B_c = np.ascontiguousarray(B, dtype=np.float32)

    if K == 0:
        return np.zeros((M, N), dtype=np.float32)

    C = np.empty((M, N), dtype=np.float32)

    a_ptr = A_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = B_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.c_gemm_f32(a_ptr, b_ptr, c_ptr, M, N, K)
    return C

def c_softmax(X: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute numerically stable Softmax along specified axis using C-engine if available,
    otherwise fallback to NumPy.
    Supports 2D arrays (and treats multi-dim flattened to 2D along last axis).
    """
    if np.iscomplexobj(X):
        raise TypeError("Complex dtypes are not supported")
    if X.ndim == 0:
        raise ValueError("Input array must have at least 1 dimension")
    if axis < -X.ndim or axis >= X.ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {X.ndim}")

    norm_axis = axis if axis >= 0 else axis + X.ndim
    if X.shape[norm_axis] == 0:
        raise ValueError("Cannot compute softmax along empty dimension")
    if X.size == 0:
        return np.empty(X.shape, dtype=X.dtype if X.dtype in (np.float32, np.float64) else np.float32)

    lib = _find_or_build_library()
    if lib is None or (axis != -1 and axis != X.ndim - 1) or X.dtype != np.float32:
        shifted = X - np.max(X, axis=axis, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=axis, keepdims=True)

    orig_shape = X.shape
    cols = orig_shape[-1]
    rows = X.size // cols
    if rows > _INT32_MAX or cols > _INT32_MAX or (rows * cols) > _INT32_MAX:
        raise ValueError(f"Dimensions exceed INT32_MAX: rows={rows}, cols={cols}")

    X_c = np.ascontiguousarray(X, dtype=np.float32).reshape(rows, cols)
    out = np.empty((rows, cols), dtype=np.float32)

    in_ptr = X_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.c_softmax_f32(in_ptr, out_ptr, rows, cols)
    return out.reshape(orig_shape)

def c_rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute RMSNorm forward pass using C-engine if available,
    otherwise fallback to pure NumPy.
    x: tensor of arbitrary shape whose last dimension is feature dimension (dim)
    weight: 1D array of shape (dim,)
    eps: numerical stability epsilon
    """
    lib = _find_or_build_library()
    if lib is None:
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        return (x / rms) * weight

    orig_shape = x.shape
    cols = orig_shape[-1]
    rows = x.size // cols

    if weight.shape != (cols,):
        raise ValueError(f"Weight shape {weight.shape} incompatible with feature dim {cols}")

    orig_dtype = x.dtype
    if orig_dtype == np.float64 or weight.dtype == np.float64:
        use_f64 = True
        x_c = np.ascontiguousarray(x, dtype=np.float64).reshape(rows, cols)
        w_c = np.ascontiguousarray(weight, dtype=np.float64)
        out = np.empty((rows, cols), dtype=np.float64)
        x_ptr = x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        w_ptr = w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.c_rmsnorm_f64(x_ptr, w_ptr, out_ptr, rows, cols, ctypes.c_double(eps))
        return out.reshape(orig_shape)
    else:
        x_c = np.ascontiguousarray(x, dtype=np.float32).reshape(rows, cols)
        w_c = np.ascontiguousarray(weight, dtype=np.float32)
        out = np.empty((rows, cols), dtype=np.float32)
        x_ptr = x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        w_ptr = w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        lib.c_rmsnorm_f32(x_ptr, w_ptr, out_ptr, rows, cols, ctypes.c_float(eps))
        return out.reshape(orig_shape)

