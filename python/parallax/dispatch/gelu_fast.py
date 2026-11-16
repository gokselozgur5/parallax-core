"""GeLU fast approximation and tanh forward dispatch operations."""

from __future__ import annotations

import math
from typing import Any, Optional
import numpy as np

from parallax.nn import Module
from parallax.tensor import Tensor, _accumulate_grad

SQRT_2_OVER_PI: float = 0.7978845608028654
COEF: float = 0.044715


def gelu_fast_forward_np(x: np.ndarray) -> np.ndarray:
    """NumPy reference implementation of fast GeLU tanh approximation."""
    if np.iscomplexobj(x):
        raise TypeError("Complex dtypes are not supported")
    x_f32 = np.asarray(x, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        inner = SQRT_2_OVER_PI * (x_f32 + COEF * (x_f32**3))
        res = 0.5 * x_f32 * (1.0 + np.tanh(inner))
    res = np.where(x_f32 <= -10.0, 0.0, res)
    res = np.where(x_f32 >= 10.0, x_f32, res)
    return res.astype(x.dtype)


def gelu_fast_dispatch(x: np.ndarray) -> np.ndarray:
    """Compute fast GeLU activation via backend engine, native C, or NumPy."""
    if np.iscomplexobj(x):
        raise TypeError("Complex dtypes are not supported")
    try:
        from parallax.backends.dispatch import dispatch
        return dispatch("gelu_fast", x)
    except Exception:
        pass
    try:
        from parallax.native.gelu import c_gelu
        return c_gelu(x)
    except Exception:
        pass
    return gelu_fast_forward_np(x)
