from __future__ import annotations

import ctypes

import numpy as np

from parallax.native.loader import bind

_FP = ctypes.POINTER(ctypes.c_float)
_c_gqa = bind("c_gqa_proj_f32", [_FP] * 8 + [ctypes.c_int] * 5, ctypes.c_int)


def is_gqa_proj_available() -> bool:
    """Return True if native c_gqa_proj_f32 is bound."""
