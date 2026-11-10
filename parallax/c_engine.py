"""
Parallax C-Engine: High-performance C implementations for GEMM and Softmax with ctypes and NumPy fallback.
"""

from __future__ import annotations

import ctypes
import os
import sys
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

def _find_or_build_library() -> Optional[ctypes.CDLL]:
    global _C_LIB, _LIB_LOAD_ATTEMPTED
    if _LIB_LOAD_ATTEMPTED:
        return _C_LIB

    _LIB_LOAD_ATTEMPTED = True
    csrc_dir = Path(__file__).parent / "csrc"
    
    # Candidate shared library names
    exts = [".dylib", ".so", ".dll"]
    candidate_names = [f"libparallax_c{ext}" for ext in exts]
    
    lib_path = None
    for name in candidate_names:
        candidate = csrc_dir / name
        if candidate.is_file():
            lib_path = candidate
            break

    # If not found, attempt compilation if Makefile and compiler exist
    if lib_path is None and (csrc_dir / "Makefile").is_file():
        try:
            subprocess.run(["make", "-C", str(csrc_dir)], check=True, capture_output=True)
            for name in candidate_names:
                candidate = csrc_dir / name
                if candidate.is_file():
                    lib_path = candidate
                    break
        except Exception:
            pass

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
        except Exception:
            _C_LIB = None

    return _C_LIB

def is_c_engine_available() -> bool:
    """Return True if the native C-engine library is compiled and loaded."""
    return _find_or_build_library() is not None

def c_gemm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute matrix multiplication C = A @ B using C-engine if available,
    otherwise fallback to NumPy.
    A: 2D array of shape (M, K), float32
    B: 2D array of shape (K, N), float32
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"Inputs must be 2D matrices, got shapes {A.shape} and {B.shape}")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible shapes for GEMM: {A.shape} and {B.shape}")

    lib = _find_or_build_library()
    if lib is None:
        return np.matmul(A, B, dtype=np.float32)

    A_c = np.ascontiguousarray(A, dtype=np.float32)
    B_c = np.ascontiguousarray(B, dtype=np.float32)
    M, K = A_c.shape
    K2, N = B_c.shape

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
    lib = _find_or_build_library()
    
    if lib is None or (axis != -1 and axis != X.ndim - 1):
        # Fallback to pure numpy
        shifted = X - np.max(X, axis=axis, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=axis, keepdims=True)

    orig_shape = X.shape
    cols = orig_shape[-1]
    rows = X.size // cols

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

