"""AVX-512 SGEMM microkernel binding."""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional
import numpy as np
from parallax.native.loader import bind

_FN_AVX512: Optional[Callable[..., Any]] = None
_FN_HAS_AVX512: Optional[Callable[..., Any]] = None


def _get_avx512_fn() -> Optional[Callable[..., Any]]:
    global _FN_AVX512
    if _FN_AVX512 is None:
        args = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
        _FN_AVX512 = bind("c_gemm_avx512_f32", args, ctypes.c_int)
    return _FN_AVX512


def is_avx512_available() -> bool:
    """Return True if AVX-512 is supported by the CPU."""
    global _FN_HAS_AVX512
    if _FN_HAS_AVX512 is None:
        _FN_HAS_AVX512 = bind("c_has_avx512_f32", [], ctypes.c_int)
    return bool(_FN_HAS_AVX512() != 0) if _FN_HAS_AVX512 else False


def gemm_avx512(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Execute matrix multiplication C = A @ B using AVX-512."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    fn = _get_avx512_fn()
    if fn is None:
        raise RuntimeError("c_gemm_avx512_f32 symbol not available")
    m, k = a.shape
    _, n = b.shape
    a_c = np.ascontiguousarray(a, dtype=np.float32)
    b_c = np.ascontiguousarray(b, dtype=np.float32)
    c = np.empty((m, n), dtype=np.float32)
    status = fn(a_c.ctypes.data, b_c.ctypes.data, c.ctypes.data, m, n, k)
    if status == -5:
        raise RuntimeError("AVX-512F is not supported on this CPU")
    if status != 0:
        raise RuntimeError(f"c_gemm_avx512_f32 failed with status {status}")
    return c

