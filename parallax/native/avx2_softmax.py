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
