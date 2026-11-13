"""Python wrapper for the NEON polynomial exp softmax microkernel."""

from __future__ import annotations

import ctypes

import numpy as np

from parallax.native.loader import bind

_c_softmax_neon_f32 = bind(
    "c_softmax_neon_f32",
    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
    ctypes.c_int,
)


def is_softmax_neon_available() -> bool:
    """Return True if the NEON softmax symbol is bound."""
    return _c_softmax_neon_f32 is not None


def softmax_neon(x: np.ndarray) -> np.ndarray:
    """Compute softmax along the last dimension using NEON."""
    if _c_softmax_neon_f32 is None:
        raise RuntimeError("NEON softmax microkernel is not available")
    if x.ndim != 2:
        raise ValueError("Input must be a 2D array")
    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    rows, cols = x_f32.shape
    out = np.empty((rows, cols), dtype=np.float32)
    status = _c_softmax_neon_f32(
        x_f32.ctypes.data_as(ctypes.c_void_p),
        out.ctypes.data_as(ctypes.c_void_p),
        rows,
        cols,
    )
    if status != 0:
        raise RuntimeError(f"c_softmax_neon_f32 failed: {status}")
    return out
