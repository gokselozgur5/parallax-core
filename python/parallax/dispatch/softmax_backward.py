"""Dispatched Softmax backward Jacobian vector product kernel."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from parallax.nn import Module
from parallax.tensor import Tensor, _accumulate_grad


def softmax_forward_reference(
    x: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """Compute numerically stable softmax probabilities along axis."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    denom = np.sum(exp_shifted, axis=axis, keepdims=True)
    return (exp_shifted / denom).astype(x.dtype)


def softmax_backward_reference(
    dout: np.ndarray,
    probs: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """Compute analytical backward gradient via vector-Jacobian product."""
    sum_dout_probs = np.sum(dout * probs, axis=axis, keepdims=True)
    dx = probs * (dout - sum_dout_probs)
    return dx.astype(dout.dtype)


def softmax_backward_jvp(
    probs: np.ndarray,
    v: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """Evaluate Jacobian-vector product J(p) * v for softmax."""
    v_dot_p = np.sum(v * probs, axis=axis, keepdims=True)
    return (probs * (v - v_dot_p)).astype(v.dtype)


def softmax_backward_from_logits(
    dout: np.ndarray,
    x: np.ndarray,
    axis: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute softmax probabilities and input gradient directly from logits."""
    probs = softmax_forward_reference(x, axis=axis)
    dx = softmax_backward_reference(dout, probs, axis=axis)
    return probs, dx


def fused_masked_softmax_backward(
    dout: np.ndarray,
    probs: np.ndarray,
    mask: Optional[np.ndarray] = None,
    axis: int = -1,
) -> np.ndarray:
    """Compute softmax backward pass with optional boolean gradient mask."""
    effective_dout = dout
    if mask is not None:
        if mask.dtype == bool:
            effective_dout = np.where(mask, 0.0, dout)
        else:
            effective_dout = dout * mask
    return softmax_backward_reference(effective_dout, probs, axis=axis)


def _dispatch_softmax_backward(
    dout: np.ndarray,
    probs: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """Route softmax backward through backend dispatch or fallback."""
    try:
        from parallax.backends.dispatch import dispatch
        return dispatch("softmax_backward", dout, probs, axis=axis)
    except Exception:
        return softmax_backward_reference(dout, probs, axis=axis)


def softmax_backward(
    dout: np.ndarray,
    probs: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """Execute functional backward pass for Softmax layer."""
    return _dispatch_softmax_backward(dout, probs, axis=axis)
