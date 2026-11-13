"""AVX2/FMA SGEMM microkernel bindings."""

from __future__ import annotations

import ctypes
from typing import Any, Optional
import numpy as np

from parallax.native.loader import bind

_c_gemm_avx2 = bind(
    "c_gemm_avx2_f32",
    [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    restype=None,
)


def is_avx2_gemm_available() -> bool:
    """Return True if c_gemm_avx2_f32 symbol is available."""
    return _c_gemm_avx2 is not None
