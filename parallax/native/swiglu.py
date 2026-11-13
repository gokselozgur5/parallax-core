"""Python wrapper for the native C SwiGLU kernel."""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional
import numpy as np

from parallax.native.loader import bind

_SWIGLU_FN: Optional[Callable[..., Any]] = None
_SWIGLU_INIT_ATTEMPTED: bool = False


def _get_swiglu_fn() -> Optional[Callable[..., Any]]:
    global _SWIGLU_FN, _SWIGLU_INIT_ATTEMPTED
    if not _SWIGLU_INIT_ATTEMPTED:
        _SWIGLU_FN = bind(
            "c_swiglu_f32",
            [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ],
            None,
        )
        _SWIGLU_INIT_ATTEMPTED = True
    return _SWIGLU_FN


def swiglu_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """NumPy reference implementation of SwiGLU: silu(a) * b."""
    a_f32 = np.asarray(a, dtype=np.float32)
    b_f32 = np.asarray(b, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        sig = np.float32(1.0) / (np.float32(1.0) + np.exp(-a_f32))
        res = a_f32 * sig * b_f32
    res = np.where(np.isnan(res), np.float32(0.0), res)
    return res


def c_swiglu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute fused SwiGLU (silu(a) * b) using native C engine or NumPy."""
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        raise TypeError("Complex dtypes are not supported")
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    a_f32 = np.ascontiguousarray(a, dtype=np.float32)
    b_f32 = np.ascontiguousarray(b, dtype=np.float32)
    n = a_f32.size
    if n == 0:
        return np.empty_like(a_f32)

    fn = _get_swiglu_fn()
    if fn is None:
        return swiglu_numpy(a_f32, b_f32)

    out = np.empty_like(a_f32)
    a_ptr = a_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    fn(a_ptr, b_ptr, out_ptr, n)
    return out
