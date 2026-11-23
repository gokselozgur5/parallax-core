"""Python wrapper for the native C SwiGLU kernel."""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional
import numpy as np

from parallax.native.loader import check_int32_bounds, lazy_bind

_SWIGLU_SYM = lazy_bind(
    "c_swiglu_f32",
    [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int],
    None,
)


def _get_swiglu_fn() -> Optional[Callable[..., Any]]:
    return _SWIGLU_SYM.resolve()


def swiglu_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """NumPy reference implementation of SwiGLU: silu(a) * b."""
    a_f32 = np.asarray(a, dtype=np.float32)
    b_f32 = np.asarray(b, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        sig = np.float32(1.0) / (np.float32(1.0) + np.exp(-a_f32))
        silu = a_f32 * sig
        silu = np.where(a_f32 <= np.float32(-88.0), np.float32(0.0), silu)
        silu = np.where(a_f32 >= np.float32(88.0), a_f32, silu)
        res = silu * b_f32
    return res


def c_swiglu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute fused SwiGLU (silu(a) * b) using native C engine or NumPy."""
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        raise TypeError("Complex dtypes are not supported")
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    check_int32_bounds(int(a.size), label="Input size")

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
