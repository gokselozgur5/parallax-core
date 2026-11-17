"""Fused GeLU backward functional kernels with tanh derivative support."""

from __future__ import annotations

import math
from typing import Any
import numpy as np

SQRT_2_OVER_PI: float = 0.7978845608028654
COEF: float = 0.044715
SQRT_2: float = 1.4142135623730951
SQRT_2_PI: float = 2.5066282746310002


def gelu_backward_reference(
    dout: np.ndarray,
    x: np.ndarray,
    approximate: str = "tanh",
) -> np.ndarray:
    """Compute analytical backward pass gradient for GeLU activation."""
    if np.iscomplexobj(x) or np.iscomplexobj(dout):
        raise TypeError("Complex dtypes are not supported")
    if approximate not in {"tanh", "none"}:
        raise ValueError(f"Unsupported mode: {approximate!r}")
    x_f32 = np.asarray(x, dtype=np.float32)
    g_f32 = np.asarray(dout, dtype=np.float32)
    if approximate == "tanh":
        with np.errstate(over="ignore", invalid="ignore"):
            inner = SQRT_2_OVER_PI * (x_f32 + COEF * (x_f32**3))
            t = np.tanh(inner)
            d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * COEF * (x_f32**2))
            d_cdf = 0.5 * (1.0 - t**2) * d_inner
            dx = g_f32 * (0.5 * (1.0 + t) + x_f32 * d_cdf)
        dx = np.where(x_f32 <= -10.0, 0.0, dx)
        dx = np.where(x_f32 >= 10.0, g_f32, dx)
        return dx.astype(x.dtype)
    vec_erf = np.vectorize(math.erf, otypes=[np.float32])
    cdf = 0.5 * (1.0 + vec_erf(x_f32 / SQRT_2))
    pdf = np.exp(-0.5 * (x_f32**2)) / SQRT_2_PI
    dx = g_f32 * (cdf + x_f32 * pdf)
    return dx.astype(x.dtype)


def fused_gelu_backward(
    dout: np.ndarray,
    x: np.ndarray,
    tanh_out: np.ndarray | None = None,
    approximate: str = "tanh",
) -> np.ndarray:
    """Compute fused GeLU backward reusing cached tanh activation."""
    if approximate == "none":
        return gelu_backward_reference(dout, x, approximate="none")
    if np.iscomplexobj(x) or np.iscomplexobj(dout):
        raise TypeError("Complex dtypes are not supported")
    x_f32 = np.asarray(x, dtype=np.float32)
    g_f32 = np.asarray(dout, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        if tanh_out is not None:
            t = np.asarray(tanh_out, dtype=np.float32)
        else:
            inner = SQRT_2_OVER_PI * (x_f32 + COEF * (x_f32**3))
            t = np.tanh(inner)
        d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * COEF * (x_f32**2))
        d_cdf = 0.5 * (1.0 - t**2) * d_inner
        dx = g_f32 * (0.5 * (1.0 + t) + x_f32 * d_cdf)
    dx = np.where(x_f32 <= -10.0, 0.0, dx)
    dx = np.where(x_f32 >= 10.0, g_f32, dx)
    return dx.astype(x.dtype)
