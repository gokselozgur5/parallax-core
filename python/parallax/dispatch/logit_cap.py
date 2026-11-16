"""Logit soft-capping and tanh-capping dispatch operations."""

from __future__ import annotations

from typing import Any, Optional
import numpy as np


def _dispatch_logit_cap(x: np.ndarray, cap: float) -> np.ndarray:
    """Route logit soft capping through backend dispatch engine if available."""
    try:
        from parallax.backends.dispatch import dispatch
        return dispatch("logit_cap", x, cap=cap)
    except Exception:
        scaled = x / cap
        return (cap * np.tanh(scaled)).astype(x.dtype)


def logit_soft_cap(x: np.ndarray, cap: float = 30.0) -> np.ndarray:
    """Apply hyperbolic tangent soft-capping to logits: cap * tanh(x / cap)."""
    cap_val = float(cap)
    if cap_val <= 0.0:
        raise ValueError("cap must be strictly positive")
    return _dispatch_logit_cap(x, cap_val)



def logit_soft_cap_backward(
    grad_out: np.ndarray,
    x: np.ndarray,
    cap: float = 30.0,
) -> np.ndarray:
    """Compute analytical backward pass gradient for logit soft capping."""
    cap_val = float(cap)
    if cap_val <= 0.0:
        raise ValueError("cap must be strictly positive")
    t = np.tanh(x / cap_val)
    grad_x = grad_out * (1.0 - t * t)
    return grad_x.astype(x.dtype)


