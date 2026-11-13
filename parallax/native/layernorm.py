from __future__ import annotations

import ctypes
from typing import Optional
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


def c_layernorm(
    x: np.ndarray,
    weight: Optional[np.ndarray] = None,
    bias: Optional[np.ndarray] = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """Apply LayerNorm on float32 array using C engine."""
    if x.dtype != np.float32:
        raise TypeError("x must be float32")
    cols = x.shape[-1]
    rows = x.size // cols
    x_c = np.ascontiguousarray(x)
    out = np.empty_like(x_c)
    if _c_layernorm is None:
        raise RuntimeError("c_layernorm_f32 is not available")
    w_c = np.ascontiguousarray(weight, dtype=np.float32) if weight is not None else None
    b_c = np.ascontiguousarray(bias, dtype=np.float32) if bias is not None else None
    w_p = w_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if w_c is not None else None
    b_p = b_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if b_c is not None else None
    x_p = x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_p = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    _c_layernorm(x_p, w_p, b_p, out_p, rows, cols, float(eps))
    return out
