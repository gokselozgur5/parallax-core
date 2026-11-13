from __future__ import annotations

import ctypes
import numpy as np

from parallax.native.loader import bind

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
