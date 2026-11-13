"""AVX2 RMSNorm kernel bindings."""

from __future__ import annotations

import ctypes

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


def rmsnorm_avx2(
    x: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute RMSNorm using the AVX2 C kernel."""
    if not is_avx2_rmsnorm_available():
        raise RuntimeError("AVX2 RMSNorm kernel is unavailable")
    if x.ndim != 2 or weight.ndim != 1 or x.shape[1] != weight.shape[0]:
        raise ValueError(f"shape mismatch: x={x.shape}, weight={weight.shape}")
    x_c = np.ascontiguousarray(x, dtype=np.float32)
    w_c = np.ascontiguousarray(weight, dtype=np.float32)
    out = np.empty_like(x_c)
    rows, cols = x_c.shape
    _rmsnorm_avx2(  # type: ignore[misc]
        x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rows,
        cols,
        ctypes.c_float(eps),
    )
    return out


__all__ = ["is_avx2_rmsnorm_available", "rmsnorm_avx2"]
