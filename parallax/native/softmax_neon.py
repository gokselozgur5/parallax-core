"""Python wrapper for the NEON polynomial exp softmax microkernel."""

from __future__ import annotations

import ctypes

import numpy as np

from parallax.native.loader import check_int32_bounds, lazy_bind

_c_softmax_neon_f32 = lazy_bind(
    "c_softmax_neon_f32",
    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
    ctypes.c_int,
)


def is_softmax_neon_available() -> bool:
    """Return True if the NEON softmax symbol is bound."""
    return bool(_c_softmax_neon_f32)


def softmax_neon(x: np.ndarray) -> np.ndarray:
    """Compute softmax along the last dimension using NEON."""
    if not _c_softmax_neon_f32:
        raise RuntimeError("NEON softmax microkernel is not available")
    if x.ndim != 2:
        raise ValueError("Input must be a 2D array")
    rows, cols = x.shape
    check_int32_bounds(rows, cols, rows * cols)
    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
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
