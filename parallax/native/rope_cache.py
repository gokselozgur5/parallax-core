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

    @classmethod
    def create(cls, dim: int, max_seq_len: int, theta: float = 10000.0) -> RoPECache:
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"dim must be positive and even, got {dim}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if theta <= 0.0:
            raise ValueError(f"theta must be positive, got {theta}")
        half = dim // 2
        cos = np.empty((max_seq_len, half), dtype=np.float32)
        sin = np.empty((max_seq_len, half), dtype=np.float32)
