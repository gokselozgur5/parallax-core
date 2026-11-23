from __future__ import annotations

import ctypes
import numpy as np

from parallax.native.loader import INT32_MAX, bind

_c_rope = bind(
    "c_rope_f32",
    [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
    ],
    None,
)


def is_rope_available() -> bool:
    """Return True if native c_rope_f32 is bound and callable."""
    return _c_rope is not None


def rope_in_place(
    x: np.ndarray,
    start_pos: int = 0,
    theta: float = 10000.0,
    interleaved: bool = True,
) -> np.ndarray:
    """Apply rotary positional embedding in place on [T, H, D] float32 array."""
    if x.ndim != 3:
        raise ValueError(f"x must have shape [T, H, D], got ndim {x.ndim}")
    if x.shape[-1] % 2 != 0:
        raise ValueError("last dimension must be even")
    if x.dtype != np.float32:
        raise TypeError("x must be float32")
    T, H, D = x.shape
    if max(T, H, D, T * H * D) > INT32_MAX or not (-INT32_MAX - 1 <= start_pos <= INT32_MAX):
        raise ValueError(f"Dimensions exceed INT32_MAX: shape={x.shape}, start_pos={start_pos}")
    if not x.flags.c_contiguous:
        raise ValueError("x must be C-contiguous")
    if _c_rope is None:
        raise RuntimeError("c_rope_f32 is not available")

    ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    _c_rope(ptr, T, H, D, start_pos, float(theta), 1 if interleaved else 0)
    return x
