"""AVX2 softmax microkernel bindings."""

from __future__ import annotations

import ctypes
import numpy as np

from parallax.native.loader import bind

_c_softmax_avx2 = bind(
    "c_softmax_avx2_f32",
    [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ],
    restype=None,
)


def is_avx2_softmax_available() -> bool:
    """Return True if c_softmax_avx2_f32 symbol is available."""
    return _c_softmax_avx2 is not None


def softmax_avx2(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute row-wise softmax via c_softmax_avx2_f32."""
    if _c_softmax_avx2 is None:
        raise RuntimeError("c_softmax_avx2_f32 symbol is not available")
    if axis not in (-1, x.ndim - 1):
        raise ValueError("softmax_avx2 only supports last dimension")
    if x.ndim != 2:
        raise ValueError("input must be a 2D matrix")

    rows, cols = x.shape
    x_c = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty_like(x_c)

    in_ptr = x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    _c_softmax_avx2(in_ptr, out_ptr, rows, cols)
    return out
