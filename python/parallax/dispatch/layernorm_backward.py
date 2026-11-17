"""LayerNorm backward autograd functional kernel with dispatch support."""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np
from parallax.tensor import Tensor, _accumulate_grad


def stabilize_epsilon(eps: float, min_eps: float = 1e-12) -> float:
    """Stabilize epsilon ensuring strictly positive bounded threshold."""
    return max(float(eps), min_eps)


def layernorm_backward_reference(
    dout: np.ndarray,
    x: np.ndarray,
    weight: Optional[np.ndarray] = None,
    bias: Optional[np.ndarray] = None,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute analytical input, weight, and bias gradients for LayerNorm."""
    eps_val = stabilize_epsilon(eps)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps_val)
    x_hat = (x - mean) * inv_std
    gw = dout * weight if weight is not None else dout
    dx = inv_std * (gw - np.mean(gw, axis=-1, keepdims=True) - x_hat * np.mean(gw * x_hat, axis=-1, keepdims=True))
    reduce_axes = tuple(range(dout.ndim - 1)) if dout.ndim > 1 else None
    dw = np.sum(dout * x_hat, axis=reduce_axes) if weight is not None else None
    db = np.sum(dout, axis=reduce_axes) if bias is not None else None
    return dx.astype(x.dtype), dw if dw is None else dw.astype(weight.dtype), db if db is None else db.astype(bias.dtype)


def _dispatch_layernorm_backward(
    dout: np.ndarray, x: np.ndarray, weight: Optional[np.ndarray], bias: Optional[np.ndarray], eps: float
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        from parallax.backends.dispatch import dispatch
        return dispatch("layernorm_backward", dout, x, weight=weight, bias=bias, eps=eps)
    except Exception:
        return layernorm_backward_reference(dout, x, weight=weight, bias=bias, eps=eps)


def layernorm_backward(
    dout: np.ndarray, x: np.ndarray, weight: Optional[np.ndarray] = None,
    bias: Optional[np.ndarray] = None, eps: float = 1e-5
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Execute functional backward pass for LayerNorm."""
    return _dispatch_layernorm_backward(dout, x, weight, bias, stabilize_epsilon(eps))


def fused_layernorm_backward(
    dout: np.ndarray,
    x: np.ndarray,
    weight: Optional[np.ndarray] = None,
    bias: Optional[np.ndarray] = None,
    residual: Optional[np.ndarray] = None,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute gradients for LayerNorm with optional fused residual."""
    fused = x + residual if residual is not None else x
    dx, dw, db = layernorm_backward(dout, fused, weight, bias, eps=eps)
    dres = dx.copy() if residual is not None else None
    return dx, dw, db, dres
