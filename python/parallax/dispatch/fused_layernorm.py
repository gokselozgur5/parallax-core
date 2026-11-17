"""Dispatched LayerNorm with RMSNorm weight scaling and bias fusion."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import numpy as np

from parallax.nn import Module, Parameter
from parallax.tensor import Tensor, _accumulate_grad


def stabilize_epsilon(eps: float, min_eps: float = 1e-12) -> float:
    """Stabilize epsilon parameter ensuring strictly positive bounded value."""
    return max(float(eps), min_eps)


def rms_weight_scale(weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Scale weight tensor by its root-mean-square norm along the last axis."""
    eps_val = stabilize_epsilon(eps)
    rms = np.sqrt(np.mean(weight * weight, axis=-1, keepdims=True) + eps_val)
    return weight / rms


def rms_weight_scale_backward(
    dout: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute gradient with respect to original unscaled weight."""
    eps_val = stabilize_epsilon(eps)
    mean_sq = np.mean(weight * weight, axis=-1, keepdims=True)
    rms = np.sqrt(mean_sq + eps_val)
    w_hat = weight / rms
    dot = np.mean(dout * w_hat, axis=-1, keepdims=True)
    return (dout - w_hat * dot) / rms


def _dispatch_fused_layernorm(
    x: np.ndarray,
    weight: Optional[np.ndarray],
    bias: Optional[np.ndarray],
    eps: float,
) -> np.ndarray:
    """Route fused layernorm to backend dispatcher or fallback to numpy."""
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
