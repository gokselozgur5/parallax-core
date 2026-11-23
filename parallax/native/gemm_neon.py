"""Python wrapper for the NEON 8x8 SGEMM microkernel."""

from __future__ import annotations

import ctypes

import numpy as np

from parallax.native.loader import check_int32_bounds, lazy_bind

_c_gemm_neon_f32 = lazy_bind(
    "c_gemm_neon_f32",
    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ctypes.c_int,
)


def is_gemm_neon_available() -> bool:
    """Return True if the NEON SGEMM symbol is bound."""
    return bool(_c_gemm_neon_f32)


def gemm_neon(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute matrix multiplication using the NEON microkernel."""
    if not _c_gemm_neon_f32:
        raise RuntimeError("NEON GEMM microkernel is not available")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Invalid array dimensions for gemm_neon")
    m, k = a.shape
    _, n = b.shape
    check_int32_bounds(m, n, k, m * k, k * n, m * n, label="Matrix dimensions")
    a_f32 = np.ascontiguousarray(a, dtype=np.float32)
    b_f32 = np.ascontiguousarray(b, dtype=np.float32)
    out = np.empty((m, n), dtype=np.float32)
    status = _c_gemm_neon_f32(
        a_f32.ctypes.data_as(ctypes.c_void_p),
        b_f32.ctypes.data_as(ctypes.c_void_p),
        out.ctypes.data_as(ctypes.c_void_p),
        m, n, k,
    )
    if status != 0:
        raise RuntimeError(f"c_gemm_neon_f32 failed: {status}")
    return out
