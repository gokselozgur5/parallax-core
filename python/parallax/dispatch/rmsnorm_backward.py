"""RMSNorm backward autograd functional kernel with dispatch support."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import numpy as np


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
