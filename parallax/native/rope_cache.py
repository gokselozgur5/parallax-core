from __future__ import annotations
import ctypes
from dataclasses import dataclass
import numpy as np
from parallax.native.loader import INT32_MAX, bind

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
        if max(dim, max_seq_len, max_seq_len * half) > INT32_MAX:
            raise ValueError(f"Dimensions exceed INT32_MAX: max_seq_len={max_seq_len}, dim={dim}")
        cos = np.empty((max_seq_len, half), dtype=np.float32)
        sin = np.empty((max_seq_len, half), dtype=np.float32)
        if _c_build_rope_cache is not None:
            c_ptr = cos.ctypes.data_as(_f32_ptr)
            s_ptr = sin.ctypes.data_as(_f32_ptr)
            rc = _c_build_rope_cache(c_ptr, s_ptr, int(max_seq_len), int(dim), float(theta))
            if rc != 0:
                raise RuntimeError(f"c_build_rope_cache_f32 failed: {rc}")
        else:
            freq = np.arange(half, dtype=np.float64)
            inv_freq = 1.0 / (float(theta) ** (freq / half))
            pos = np.arange(max_seq_len, dtype=np.float64)[:, None]
            ang = pos * inv_freq[None, :]
            cos[:] = np.cos(ang).astype(np.float32)
            sin[:] = np.sin(ang).astype(np.float32)
        return cls(cos=cos, sin=sin, dim=dim, max_seq_len=max_seq_len, theta=float(theta))

    def apply_in_place(
        self, x: np.ndarray, start_pos: int = 0, interleaved: bool = True
    ) -> np.ndarray:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [T, H, D], got ndim {x.ndim}")
        if x.dtype != np.float32:
            raise TypeError("x must be float32")
        T, H, D = x.shape
        if D != self.dim:
            raise ValueError(f"dim mismatch: tensor {D} vs cache {self.dim}")
        if max(T, H, D, T * H * D, self.max_seq_len) > INT32_MAX:
            raise ValueError(f"Dimensions exceed INT32_MAX: shape={x.shape}")
        if start_pos < 0 or start_pos + T > self.max_seq_len:
            raise ValueError(f"range [{start_pos}, {start_pos + T}) invalid")
        if not x.flags.c_contiguous:
            raise ValueError("x must be C-contiguous")
        if T == 0 or H == 0 or D == 0:
            return x
        if _c_apply_rope_cached is not None:
            xp = x.ctypes.data_as(_f32_ptr)
            cp = self.cos.ctypes.data_as(_f32_ptr)
            sp = self.sin.ctypes.data_as(_f32_ptr)
            rc = _c_apply_rope_cached(
                xp, cp, sp, int(T), int(H), int(D),
                int(start_pos), int(self.max_seq_len), 1 if interleaved else 0,
            )
            if rc != 0:
                raise RuntimeError(f"c_apply_rope_cached_f32 failed: {rc}")
            return x
        cos = self.cos[start_pos : start_pos + T, None, :]
        sin = self.sin[start_pos : start_pos + T, None, :]
        if interleaved:
            x1 = x[..., 0::2]
            x2 = x[..., 1::2]
            rx1 = x1 * cos - x2 * sin
            rx2 = x1 * sin + x2 * cos
            x[..., 0::2] = rx1
            x[..., 1::2] = rx2
        else:
            half = D // 2
            x1 = x[..., :half]
            x2 = x[..., half:]
            rx1 = x1 * cos - x2 * sin
            rx2 = x1 * sin + x2 * cos
            x[..., :half] = rx1
            x[..., half:] = rx2
        return x
