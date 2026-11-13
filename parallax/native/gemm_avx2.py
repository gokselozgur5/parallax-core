"""AVX2/FMA SGEMM microkernel bindings."""

from __future__ import annotations

import ctypes
from typing import Any, Optional
import numpy as np

from parallax.native.loader import bind

_c_gemm_avx2 = bind(
    "c_gemm_avx2_f32",
    [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    restype=None,
)


def is_avx2_gemm_available() -> bool:
    """Return True if c_gemm_avx2_f32 symbol is available."""
    return _c_gemm_avx2 is not None


def gemm_avx2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute dense matrix multiplication via c_gemm_avx2_f32."""
    if _c_gemm_avx2 is None:
        raise RuntimeError("c_gemm_avx2_f32 symbol is not available")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("inputs must be 2D matrices")
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    a_c = np.ascontiguousarray(a, dtype=np.float32)
    b_c = np.ascontiguousarray(b, dtype=np.float32)
    c = np.empty((m, n), dtype=np.float32)

    a_ptr = a_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_ptr = c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    _c_gemm_avx2(a_ptr, b_ptr, c_ptr, m, n, k)
    return c
