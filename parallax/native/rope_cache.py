from __future__ import annotations
import ctypes
from dataclasses import dataclass
import numpy as np
from parallax.native.loader import bind

_f32_ptr = ctypes.POINTER(ctypes.c_float)
_c_build_rope_cache = bind(
    "c_build_rope_cache_f32",
    [_f32_ptr, _f32_ptr, ctypes.c_int, ctypes.c_int, ctypes.c_float],
    ctypes.c_int,
)
_c_apply_rope_cached = bind(
    "c_apply_rope_cached_f32",
    [_f32_ptr, _f32_ptr, _f32_ptr, ctypes.c_int, ctypes.c_int,
     ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ctypes.c_int,
)
