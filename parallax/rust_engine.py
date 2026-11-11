"""Parallax Rust Engine: High-performance Rust C-ABI engine bridge with NumPy fallback.

Enforces parallax_version() == 2, pointer alignment, slice bounds checks, and
graceful fallback to NumPy kernels when the compiled Rust binary is unavailable.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import threading
from pathlib import Path
from typing import Optional
import numpy as np

_LOGGER = logging.getLogger("parallax.rust_engine")
_RUST_LIB: Optional[ctypes.CDLL] = None
_LIB_LOAD_ATTEMPTED: bool = False
_INIT_LOCK = threading.Lock()
_EXPECTED_ABI_VERSION: int = 2


def find_rust_lib() -> Optional[Path]:
    """Locate the compiled libparallax_rs library if available."""
    repo_root = Path(__file__).resolve().parent.parent
    crate_target = repo_root / "crates" / "parallax-rs" / "target"

    system = platform.system()
    if system == "Darwin":
        lib_name = "libparallax_rs.dylib"
    elif system == "Windows":
        lib_name = "parallax_rs.dll"
    else:
        lib_name = "libparallax_rs.so"

    for mode in ("release", "debug"):
        candidate = crate_target / mode / lib_name
        if candidate.is_file():
            return candidate
    return None


def get_rust_lib() -> Optional[ctypes.CDLL]:
    """Load and configure the Rust C-ABI CDLL with thread safety and ABI check."""
    global _RUST_LIB, _LIB_LOAD_ATTEMPTED
    if _LIB_LOAD_ATTEMPTED:
        return _RUST_LIB

    with _INIT_LOCK:
        if _LIB_LOAD_ATTEMPTED:
            return _RUST_LIB
        _LIB_LOAD_ATTEMPTED = True

        lib_path = find_rust_lib()
        if lib_path is None:
            _LOGGER.debug("libparallax_rs shared library not found; using fallback.")
            return None

        try:
            lib = ctypes.CDLL(str(lib_path))
            lib.parallax_version.argtypes = []
            lib.parallax_version.restype = ctypes.c_uint32
            ver = lib.parallax_version()
            if ver != _EXPECTED_ABI_VERSION:
                raise RuntimeError(
                    f"Rust library ABI version mismatch: expected {_EXPECTED_ABI_VERSION}, got {ver}. "
                    "Please rebuild with: cargo build --manifest-path crates/parallax-rs/Cargo.toml --release"
                )

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

            lib.parallax_softmax_f32.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
            ]
            lib.parallax_softmax_f32.restype = ctypes.c_int32

            _RUST_LIB = lib
            return _RUST_LIB
        except Exception as e:
            _LOGGER.warning("Failed to initialize Rust engine: %s; falling back to NumPy", e)
            _RUST_LIB = None
            if os.environ.get("PARALLAX_REQUIRE_RUST_ENGINE") == "1":
                raise
            return None


def is_rust_engine_available() -> bool:
    """Return True if the Rust engine is compiled, loaded, and matches ABI version."""
    return get_rust_lib() is not None


def rust_gemm(
    a: np.ndarray,
    b: np.ndarray,
    c: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> np.ndarray:
    """Compute C = alpha * (A @ B) + beta * C with Rust engine or NumPy fallback."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Inputs A and B must be 2-dimensional matrices")
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"Incompatible matrix dimensions: A {a.shape} vs B {b.shape}")

    if c is None:
        c_out = np.zeros((m, n), dtype=np.float32)
    else:
        if c.shape != (m, n):
            raise ValueError(f"Target buffer C shape {c.shape} does not match ({m}, {n})")
        c_out = c

    lib = get_rust_lib()
    if lib is None or a.dtype != np.float32 or b.dtype != np.float32 or c_out.dtype != np.float32:
        res = alpha * (a @ b) + (beta * c_out if beta != 0.0 else 0.0)
        c_out[...] = res
        return c_out

    a_c = np.ascontiguousarray(a, dtype=np.float32)
    b_c = np.ascontiguousarray(b, dtype=np.float32)
    c_c = np.ascontiguousarray(c_out, dtype=np.float32)

    a_ptr = a_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_ptr = c_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    status = lib.parallax_gemm_f32(
        m, n, k, alpha, a_ptr, a_c.size, b_ptr, b_c.size, beta, c_ptr, c_c.size
    )
    if status != 0:
        res = alpha * (a @ b) + (beta * c_out if beta != 0.0 else 0.0)
        c_out[...] = res
        return c_out

    if c_c is not c_out:
        c_out[...] = c_c
    return c_out


def rust_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute numerically stable Softmax using Rust engine or NumPy fallback."""
    if x.ndim == 0:
        raise ValueError("Input array must have at least 1 dimension")
    if axis < -x.ndim or axis >= x.ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {x.ndim}")

    norm_axis = axis if axis >= 0 else axis + x.ndim
    if x.shape[norm_axis] == 0:
        raise ValueError("Cannot compute softmax along empty dimension")
    if x.size == 0:
        return np.empty_like(x)

    lib = get_rust_lib()
    if lib is None or norm_axis != x.ndim - 1 or x.dtype != np.float32:
        max_s = np.max(x, axis=axis, keepdims=True)
        all_masked = np.isneginf(max_s)
        safe_max = np.where(all_masked, 0.0, max_s)
        exp_s = np.exp(x - safe_max)
        exp_s = np.where(all_masked, 0.0, exp_s)
        sum_s = np.sum(exp_s, axis=axis, keepdims=True)
        denom = np.where(sum_s == 0.0, 1.0, sum_s)
        return exp_s / denom

    orig_shape = x.shape
    cols = orig_shape[-1]
    rows = x.size // cols

    x_c = np.ascontiguousarray(x, dtype=np.float32).reshape(rows, cols).copy()
    data_ptr = x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    status = lib.parallax_softmax_f32(data_ptr, x_c.size, rows, cols)
    if status != 0:
        max_s = np.max(x, axis=axis, keepdims=True)
        all_masked = np.isneginf(max_s)
        safe_max = np.where(all_masked, 0.0, max_s)
        exp_s = np.exp(x - safe_max)
        exp_s = np.where(all_masked, 0.0, exp_s)
        sum_s = np.sum(exp_s, axis=axis, keepdims=True)
        denom = np.where(sum_s == 0.0, 1.0, sum_s)
        return exp_s / denom

    return x_c.reshape(orig_shape)
