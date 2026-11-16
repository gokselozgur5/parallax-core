"""RMSNorm backward autograd functional kernel with dispatch support."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import numpy as np
from parallax.tensor import Tensor, _accumulate_grad


def stabilize_epsilon(eps: float, min_eps: float = 1e-12) -> float:
    """Stabilize epsilon ensuring strictly positive bounded threshold."""
    return max(float(eps), min_eps)


def rmsnorm_backward_reference(
    dout: np.ndarray,
    x: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute analytical input and weight gradients for RMSNorm."""
    eps_val = stabilize_epsilon(eps)
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps_val)
    x_hat = x / rms
    d_hat = dout * weight
    dx = (1.0 / rms) * (d_hat - x_hat * np.mean(d_hat * x_hat, axis=-1, keepdims=True))
    reduce_axes = tuple(range(dout.ndim - 1))
    dw = np.sum(dout * x_hat, axis=reduce_axes) if reduce_axes else dout * x_hat
    return dx.astype(x.dtype), dw.astype(weight.dtype)


def _dispatch_rmsnorm_backward(
    dout: np.ndarray,
    x: np.ndarray,
    weight: np.ndarray,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Route RMSNorm backward through backend dispatch or fallback."""
    try:
        from parallax.backends.dispatch import dispatch
        return dispatch("rmsnorm_backward", dout, x, weight, eps=eps)
    except Exception:
        return rmsnorm_backward_reference(dout, x, weight, eps=eps)


def rmsnorm_backward(
    dout: np.ndarray,
    x: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute functional backward pass for RMSNorm layer."""
    eps_val = stabilize_epsilon(eps)
    return _dispatch_rmsnorm_backward(dout, x, weight, eps_val)


def fused_rmsnorm_backward(
    dout: np.ndarray,
    x: np.ndarray,
    weight: np.ndarray,
    residual: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Compute gradients for RMSNorm with optional fused residual."""
    fused = x + residual if residual is not None else x
    dx, dw = rmsnorm_backward(dout, fused, weight, eps=eps)
    dres = dx.copy() if residual is not None else None
    return dx, dw, dres
