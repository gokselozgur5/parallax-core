"""Python wrapper for the NEON 8x8 SGEMM microkernel."""

from __future__ import annotations

import ctypes

import numpy as np

from parallax.native.loader import bind

_c_gemm_neon_f32 = bind(
    "c_gemm_neon_f32",
    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ctypes.c_int,
)


def is_gemm_neon_available() -> bool:
    """Return True if the NEON SGEMM symbol is bound."""
    return _c_gemm_neon_f32 is not None


def gemm_neon(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute matrix multiplication using the NEON microkernel."""
    if _c_gemm_neon_f32 is None:
        raise RuntimeError("NEON GEMM microkernel is not available")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Invalid array dimensions for gemm_neon")
    a_f32 = np.ascontiguousarray(a, dtype=np.float32)
    b_f32 = np.ascontiguousarray(b, dtype=np.float32)
    m, k = a_f32.shape
    _, n = b_f32.shape
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
