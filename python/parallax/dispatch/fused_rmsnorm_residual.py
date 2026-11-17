"""Fused RMSNorm forward and backward residual kernel with dispatch support."""

from __future__ import annotations

from typing import Any, Optional, Tuple
import numpy as np

from parallax.nn import Module, Parameter
from parallax.tensor import Tensor, _accumulate_grad


def stabilize_epsilon(eps: float, min_eps: float = 1e-12) -> float:
    """Stabilize epsilon ensuring strictly positive bounded threshold."""
    return max(float(eps), min_eps)


def fused_rmsnorm_residual_forward_reference(
    x: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute analytical fused residual addition and RMSNorm forward pass."""
    eps_val = stabilize_epsilon(eps)
    fused = x + residual
    variance = np.mean(fused * fused, axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(variance + eps_val)
    normed = (fused * inv_rms) * weight
    return normed.astype(fused.dtype), fused


def fused_rmsnorm_residual_backward_reference(
    dout: np.ndarray,
    fused: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
    dresidual: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute analytical gradients for fused residual RMSNorm."""
    eps_val = stabilize_epsilon(eps)
    variance = np.mean(fused * fused, axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(variance + eps_val)
    x_hat = fused * inv_rms
    d_hat = dout * weight
    mean_term = np.mean(d_hat * x_hat, axis=-1, keepdims=True)
    dfused = inv_rms * (d_hat - x_hat * mean_term)
    if dresidual is not None:
        dfused = dfused + dresidual
    dx = dfused.astype(fused.dtype)
    dres = dfused.astype(fused.dtype)
    reduce_axes = tuple(range(dout.ndim - 1)) if dout.ndim > 1 else None
    dw = np.sum(dout * x_hat, axis=reduce_axes) if reduce_axes is not None else (dout * x_hat)
    return dx, dw.astype(weight.dtype), dres


def _dispatch_fused_rmsnorm_residual_forward(
    x: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Route fused RMSNorm forward through backend dispatch or fallback."""
    try:
        from parallax.backends.dispatch import dispatch
        return dispatch("fused_rmsnorm_residual_forward", x, residual, weight, eps=eps)
    except Exception:
        return fused_rmsnorm_residual_forward_reference(x, residual, weight, eps=eps)


def fused_rmsnorm_residual_forward(
    x: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute functional fused residual addition and RMSNorm forward pass."""
    eps_val = stabilize_epsilon(eps)
    return _dispatch_fused_rmsnorm_residual_forward(x, residual, weight, eps_val)


def _dispatch_fused_rmsnorm_residual_backward(
    dout: np.ndarray,
    fused: np.ndarray,
    weight: np.ndarray,
    eps: float,
    dresidual: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Route fused RMSNorm backward through backend dispatch or fallback."""
    try:
        from parallax.backends.dispatch import dispatch
        return dispatch(
            "fused_rmsnorm_residual_backward",
            dout,
            fused,
            weight,
            eps=eps,
            dresidual=dresidual,
        )
    except Exception:
        return fused_rmsnorm_residual_backward_reference(
            dout, fused, weight, eps=eps, dresidual=dresidual
        )


def fused_rmsnorm_residual_backward(
    dout: np.ndarray,
    fused: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
    dresidual: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute functional fused residual RMSNorm backward pass."""
    eps_val = stabilize_epsilon(eps)
    return _dispatch_fused_rmsnorm_residual_backward(
        dout, fused, weight, eps_val, dresidual=dresidual
    )
