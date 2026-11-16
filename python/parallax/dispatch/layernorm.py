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
