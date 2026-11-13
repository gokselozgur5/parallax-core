"""AVX2 RMSNorm kernel bindings."""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

from parallax.native.loader import bind

_has_avx2 = bind("c_has_avx2_rmsnorm", [], ctypes.c_int)
_rmsnorm_avx2 = bind(
    "c_rmsnorm_avx2_f32",
    [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int, ctypes.c_int, ctypes.c_float],
    None,
)


def is_avx2_rmsnorm_available() -> bool:
    """Return True if AVX2 RMSNorm kernel is supported."""
    return bool(_has_avx2 and _rmsnorm_avx2 and _has_avx2() != 0)
