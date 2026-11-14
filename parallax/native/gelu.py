"""Python wrapper for the native C GELU kernel."""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional
import numpy as np

from parallax.native.loader import bind

_GELU_FN: Optional[Callable[..., Any]] = None
_GELU_INIT_ATTEMPTED: bool = False


def _get_gelu_fn() -> Optional[Callable[..., Any]]:
    global _GELU_FN, _GELU_INIT_ATTEMPTED
    if not _GELU_INIT_ATTEMPTED:
        _GELU_FN = bind(
            "c_gelu_f32",
            [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ],
            None,
        )
        _GELU_INIT_ATTEMPTED = True
    return _GELU_FN


def gelu_numpy(x: np.ndarray) -> np.ndarray:
    """NumPy reference implementation of GELU tanh approximation."""
    x_f32 = np.asarray(x, dtype=np.float32)
    c = np.sqrt(np.float32(2.0) / np.pi, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        inner = c * (x_f32 + np.float32(0.044715) * (x_f32**3))
        res = np.float32(0.5) * x_f32 * (np.float32(1.0) + np.tanh(inner))
    res = np.where(np.isnan(res), np.float32(0.0), res)
    return res
