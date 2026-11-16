"""Fused LayerNorm dispatch with affine scaling and residual fusion."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import numpy as np

from parallax.nn import Module, Parameter
from parallax.tensor import Tensor, _accumulate_grad


def stabilize_epsilon(eps: float, min_eps: float = 1e-12) -> float:
    """Stabilize epsilon parameter ensuring strictly positive bounded value."""
    return max(float(eps), min_eps)


def _dispatch_layernorm(
    x: np.ndarray,
    weight: Optional[np.ndarray],
    bias: Optional[np.ndarray],
    eps: float,
) -> np.ndarray:
    """Route layernorm calculation through backend or fallback to numpy."""
    try:
        from parallax.backends.dispatch import dispatch
        return dispatch("layernorm", x, weight=weight, bias=bias, eps=eps)
    except Exception:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + eps)
        normed = (x - mean) * inv_std
        if weight is not None:
            normed = normed * weight
        if bias is not None:
            normed = normed + bias
        return normed


def fused_layernorm(
    x: np.ndarray,
    weight: Optional[np.ndarray] = None,
    bias: Optional[np.ndarray] = None,
    residual: Optional[np.ndarray] = None,
    eps: float = 1e-5,
    return_residual: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Compute LayerNorm with optional fused residual and affine parameters."""
    eps_val = stabilize_epsilon(eps)
    fused = x + residual if residual is not None else x
    orig_dtype = fused.dtype
    acc = fused.astype(np.float32) if orig_dtype in (np.float16, np.int32) else fused
    w_acc = weight.astype(acc.dtype) if weight is not None else None
    b_acc = bias.astype(acc.dtype) if bias is not None else None
    normed = _dispatch_layernorm(acc, w_acc, b_acc, eps_val)
    out = normed.astype(orig_dtype)
    if return_residual:
        return out, fused
    return out


def _backward_layernorm(
    dout: np.ndarray,
    fused: np.ndarray,
    eps: float,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    x: Optional[Tensor],
    residual: Optional[Tensor],
) -> None:
    """Compute and accumulate gradients for LayerNorm operation."""
    mean = np.mean(fused, axis=-1, keepdims=True)
    var = np.var(fused, axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (fused - mean) * inv_std
    d = fused.shape[-1]
    gw = dout * weight.data if weight is not None else dout
    dvar = np.sum(gw * (fused - mean) * -0.5 * (inv_std ** 3), axis=-1, keepdims=True)
    dmean = np.sum(gw * -inv_std, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * (fused - mean), axis=-1, keepdims=True)
    dx = gw * inv_std + dvar * 2.0 * (fused - mean) / d + dmean / d
    if weight is not None and weight.requires_grad:
        _accumulate_grad(weight, np.sum(dout * x_hat, axis=tuple(range(dout.ndim - 1))))
    if bias is not None and bias.requires_grad:
        _accumulate_grad(bias, np.sum(dout, axis=tuple(range(dout.ndim - 1))))
    if isinstance(x, Tensor) and x.requires_grad:
        _accumulate_grad(x, dx)
    if isinstance(residual, Tensor) and residual.requires_grad:
        _accumulate_grad(residual, dx)
