"""AVX-512 softmax microkernel binding."""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional
import numpy as np
from parallax.native.loader import bind

_FN_SOFTMAX_AVX512: Optional[Callable[..., Any]] = None
_FN_HAS_AVX512_SOFTMAX: Optional[Callable[..., Any]] = None


def _get_softmax_avx512_fn() -> Optional[Callable[..., Any]]:
    global _FN_SOFTMAX_AVX512
    if _FN_SOFTMAX_AVX512 is None:
        args = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        _FN_SOFTMAX_AVX512 = bind("c_softmax_avx512_f32", args, ctypes.c_int)
    return _FN_SOFTMAX_AVX512


def is_avx512_softmax_available() -> bool:
    """Return True if AVX-512 softmax is supported by the CPU."""
    global _FN_HAS_AVX512_SOFTMAX
    if _FN_HAS_AVX512_SOFTMAX is None:
        _FN_HAS_AVX512_SOFTMAX = bind("c_has_avx512_softmax_f32", [], ctypes.c_int)
    return bool(_FN_HAS_AVX512_SOFTMAX() != 0) if _FN_HAS_AVX512_SOFTMAX else False


def softmax_avx512(x: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax along the last dimension using AVX-512."""
    if x.ndim == 0:
        raise ValueError("Input array must have at least 1 dimension")
    fn = _get_softmax_avx512_fn()
    if fn is None:
        raise RuntimeError("c_softmax_avx512_f32 symbol not available")
    orig_shape = x.shape
    cols = orig_shape[-1]
    rows = x.size // cols
    x_c = np.ascontiguousarray(x.reshape(rows, cols), dtype=np.float32)
    out = np.empty_like(x_c)
    status = fn(x_c.ctypes.data, out.ctypes.data, rows, cols)
    if status == -5:
        raise RuntimeError("AVX-512F is not supported on this CPU")
    if status != 0:
        raise RuntimeError(f"c_softmax_avx512_f32 failed with status {status}")
    return out.reshape(orig_shape)

