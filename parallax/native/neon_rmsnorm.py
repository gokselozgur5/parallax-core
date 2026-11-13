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


def rmsnorm_neon(
    x: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute RMSNorm using the NEON C kernel."""
    if not is_neon_rmsnorm_available():
        raise RuntimeError("NEON RMSNorm kernel is unavailable")
    if x.ndim != 2 or weight.ndim != 1 or x.shape[1] != weight.shape[0]:
        raise ValueError(f"shape mismatch: x={x.shape}, weight={weight.shape}")
    x_c = np.ascontiguousarray(x, dtype=np.float32)
    w_c = np.ascontiguousarray(weight, dtype=np.float32)
    out = np.empty_like(x_c)
    rows, cols = x_c.shape
    _rmsnorm_neon(  # type: ignore[misc]
        x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rows,
        cols,
        ctypes.c_float(eps),
    )
    return out


__all__ = ["is_neon_rmsnorm_available", "rmsnorm_neon"]
