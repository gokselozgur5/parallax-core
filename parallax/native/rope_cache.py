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

def is_rope_cache_available() -> bool:
    """Return True if cached RoPE native bindings are available."""
    return _c_build_rope_cache is not None and _c_apply_rope_cached is not None


@dataclass
class RoPECache:
    """Vectorized RoPE cos/sin table precomputed up to max_seq_len."""

    cos: np.ndarray
    sin: np.ndarray
    dim: int
    max_seq_len: int
    theta: float
