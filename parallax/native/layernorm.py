from __future__ import annotations

import ctypes
import numpy as np

from parallax.native.loader import bind

_c_layernorm = bind(
    "c_layernorm_f32",
    [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ],
    None,
)


def is_layernorm_available() -> bool:
    """Return True if native c_layernorm_f32 is bound and callable."""
    return _c_layernorm is not None
