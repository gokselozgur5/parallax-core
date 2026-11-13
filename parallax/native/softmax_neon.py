"""Python wrapper for the NEON polynomial exp softmax microkernel."""

from __future__ import annotations

import ctypes

from parallax.native.loader import bind

_c_softmax_neon_f32 = bind(
    "c_softmax_neon_f32",
    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
    ctypes.c_int,
)


def is_softmax_neon_available() -> bool:
    """Return True if the NEON softmax symbol is bound."""
    return _c_softmax_neon_f32 is not None
