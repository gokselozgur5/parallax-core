"""AVX-512 RMSNorm microkernel binding."""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional
import numpy as np
from parallax.native.loader import bind

_FN_RMSNORM_AVX512: Optional[Callable[..., Any]] = None
_FN_HAS_AVX512_RMSNORM: Optional[Callable[..., Any]] = None


def is_avx512_rmsnorm_available() -> bool:
    """Return True if AVX-512 RMSNorm is supported by the CPU."""
    global _FN_HAS_AVX512_RMSNORM
    if _FN_HAS_AVX512_RMSNORM is None:
        _FN_HAS_AVX512_RMSNORM = bind("c_has_avx512_rmsnorm_f32", [], ctypes.c_int)
    return bool(_FN_HAS_AVX512_RMSNORM() != 0) if _FN_HAS_AVX512_RMSNORM else False


def _get_rmsnorm_avx512_fn() -> Optional[Callable[..., Any]]:
    global _FN_RMSNORM_AVX512
    if _FN_RMSNORM_AVX512 is None:
        args = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        _FN_RMSNORM_AVX512 = bind("c_rmsnorm_avx512_f32", args, ctypes.c_int)
    return _FN_RMSNORM_AVX512


def rmsnorm_avx512(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Compute RMSNorm using AVX-512 vectorization."""
    if not is_avx512_rmsnorm_available():
        raise RuntimeError("AVX-512 RMSNorm microkernel is unavailable")
    if x.ndim == 0:
        raise ValueError("Input array must have at least 1 dimension")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError(f"Weight shape mismatch: expected ({x.shape[-1]},), got {weight.shape}")
    fn = _get_rmsnorm_avx512_fn()
    if fn is None:
        raise RuntimeError("c_rmsnorm_avx512_f32 symbol not available")
    orig_shape = x.shape
    cols = orig_shape[-1]
    rows = x.size // cols
    x_c = np.ascontiguousarray(x.reshape(rows, cols), dtype=np.float32)
    w_c = np.ascontiguousarray(weight, dtype=np.float32)
    out = np.empty_like(x_c)
    status = fn(x_c.ctypes.data, w_c.ctypes.data, out.ctypes.data, rows, cols, float(eps))
    if status != 0:
        raise RuntimeError(f"c_rmsnorm_avx512_f32 failed with status {status}")
    return out.reshape(orig_shape)
