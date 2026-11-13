"""NEON RMSNorm kernel bindings."""

from __future__ import annotations

import ctypes

import numpy as np

from parallax.native.loader import bind

_has_neon = bind("c_has_neon_rmsnorm", [], ctypes.c_int)
_rmsnorm_neon = bind(
    "c_rmsnorm_neon_f32",
    [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int, ctypes.c_int, ctypes.c_float],
    None,
)


def is_neon_rmsnorm_available() -> bool:
    """Return True if NEON RMSNorm kernel is supported."""
    return bool(_has_neon and _rmsnorm_neon and _has_neon() != 0)
