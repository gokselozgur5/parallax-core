"""AVX2 softmax microkernel bindings."""

from __future__ import annotations

import ctypes
import numpy as np

from parallax.native.loader import check_int32_bounds, lazy_bind

_has_avx2_softmax = lazy_bind("c_has_avx2_softmax", [], ctypes.c_int)
_c_softmax_avx2 = lazy_bind(
    "c_softmax_avx2_f32",
    [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ],
    restype=ctypes.c_int,
)


def is_avx2_softmax_available() -> bool:
    """Return True if AVX2/FMA softmax kernel is supported."""
    return bool(_has_avx2_softmax and _c_softmax_avx2 and _has_avx2_softmax() != 0)


def softmax_avx2(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute row-wise softmax via c_softmax_avx2_f32."""
    if not _c_softmax_avx2:
        raise RuntimeError("c_softmax_avx2_f32 symbol is not available")
    if axis not in (-1, x.ndim - 1):
        raise ValueError("softmax_avx2 only supports last dimension")
    if x.ndim != 2:
        raise ValueError("input must be a 2D matrix")

    rows, cols = x.shape
    check_int32_bounds(rows, cols, rows * cols)
    x_c = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty_like(x_c)

    in_ptr = x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    status = _c_softmax_avx2(in_ptr, out_ptr, rows, cols)
    if status == -5:
        raise RuntimeError("AVX2/FMA is not supported on this CPU")
    if status != 0:
        raise RuntimeError(f"c_softmax_avx2_f32 failed with status {status}")
    return out
