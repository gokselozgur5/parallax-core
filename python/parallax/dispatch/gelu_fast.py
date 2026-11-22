"""GeLU fast approximation and tanh forward dispatch operations."""

from __future__ import annotations

from typing import Any
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
        res = dispatch("gelu_fast", x)
        return res.astype(x.dtype) if res.dtype != x.dtype else res
    except Exception:
        pass
    try:
        from parallax.native.gelu import c_gelu
        res = c_gelu(x)
        return res.astype(x.dtype) if res.dtype != x.dtype else res
    except Exception:
        pass
    return gelu_fast_forward_np(x)


def gelu_fast_backward(grad_out: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute analytical backward pass gradient for GeLU tanh approximation."""
    x_f32 = np.asarray(x, dtype=np.float32)
    g_f32 = np.asarray(grad_out, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        inner = SQRT_2_OVER_PI * (x_f32 + COEF * (x_f32**3))
        tanh_inner = np.tanh(inner)
        cdf = 0.5 * (1.0 + tanh_inner)
        d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * COEF * (x_f32**2))
        d_cdf = 0.5 * (1.0 - tanh_inner**2) * d_inner
        dx = g_f32 * (cdf + x_f32 * d_cdf)
    dx = np.where(x_f32 <= -10.0, 0.0, dx)
    dx = np.where(x_f32 >= 10.0, g_f32, dx)
    return dx.astype(x.dtype)


def dispatched_gelu_fast(x: Tensor | np.ndarray) -> Tensor:
    """Compute fast GeLU activation over Tensor or array with autograd tracking."""
    raw_x = x.data if isinstance(x, Tensor) else x
    out_data = gelu_fast_dispatch(raw_x)
    parents = (x,) if isinstance(x, Tensor) else ()
    out = Tensor(out_data, parents, "DispatchedGELUFast")
    if not out.requires_grad:
        return out

    def _backward() -> None:
        if out.grad is None or not isinstance(x, Tensor) or not x.requires_grad:
            return
        dx = gelu_fast_backward(out.grad, raw_x)
        _accumulate_grad(x, dx)

    out._backward = _backward
    return out


class DispatchedGELUFast(Module):
    """Fast GeLU activation module using optimal backend dispatch."""

    def __init__(self, approximate: str = "tanh") -> None:
        super().__init__()
        self.approximate = str(approximate).lower()
        if self.approximate not in {"tanh", "none"}:
            raise ValueError(
                f"Unsupported approximation mode: {approximate!r}. Must be 'tanh' or 'none'."
            )

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        return dispatched_gelu_fast(x)

    @classmethod
    def from_module(cls, mod: Any) -> DispatchedGELUFast:
        approx = getattr(mod, "approximate", "tanh")
        return cls(approximate=approx)

    def extra_repr(self) -> str:
        return f"approximate='{self.approximate}'"

    def __repr__(self) -> str:
        return f"DispatchedGELUFast({self.extra_repr()})"


GELUFast = DispatchedGELUFast
FastGELU = DispatchedGELUFast
gelu_fast = gelu_fast_dispatch
gelu_tanh = gelu_fast_dispatch


__all__ = [
    "DispatchedGELUFast",
    "FastGELU",
    "GELUFast",
    "dispatched_gelu_fast",
    "gelu_fast",
    "gelu_fast_backward",
    "gelu_fast_dispatch",
    "gelu_fast_forward_np",
    "gelu_tanh",
]
