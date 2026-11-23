"""AVX-512 SGEMM microkernel binding."""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional
import numpy as np
from parallax.native.loader import check_int32_bounds, lazy_bind

_FN_AVX512 = lazy_bind("c_gemm_avx512_f32", [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3, ctypes.c_int)
_FN_HAS_AVX512 = lazy_bind("c_has_avx512_f32", [], ctypes.c_int)


def _get_avx512_fn() -> Optional[Callable[..., Any]]:
    return _FN_AVX512.resolve()


def is_avx512_available() -> bool:
    """Return True if AVX-512 is supported by the CPU."""
    return bool(_FN_HAS_AVX512 and _FN_HAS_AVX512() != 0)


def gemm_avx512(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Execute matrix multiplication C = A @ B using AVX-512."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    m, k = a.shape
    _, n = b.shape
    check_int32_bounds(m, n, k, m * k, k * n, m * n, label="Matrix dimensions")
    fn = _get_avx512_fn()
    if fn is None:
        raise RuntimeError("c_gemm_avx512_f32 symbol not available")
    a_c = np.ascontiguousarray(a, dtype=np.float32)
    b_c = np.ascontiguousarray(b, dtype=np.float32)
    c = np.empty((m, n), dtype=np.float32)
    status = fn(a_c.ctypes.data, b_c.ctypes.data, c.ctypes.data, m, n, k)
    if status == -5:
        raise RuntimeError("AVX-512F is not supported on this CPU")
    if status != 0:
        raise RuntimeError(f"c_gemm_avx512_f32 failed with status {status}")
    return c

