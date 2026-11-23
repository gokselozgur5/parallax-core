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
    """Compute softmax backward pass; True entries in mask have their gradient zeroed."""
    effective_dout = dout
    if mask is not None:
        effective_dout = np.where(np.asarray(mask, dtype=bool), 0.0, dout).astype(dout.dtype)
    return softmax_backward_reference(effective_dout, probs, axis=axis)


def softmax_backward(
    dout: np.ndarray,
    probs: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """Execute functional backward pass for Softmax layer."""
    return softmax_backward_reference(dout, probs, axis=axis)


def dispatched_softmax(
    x: Tensor | np.ndarray,
    axis: int = -1,
) -> Tensor:
    """Dispatched Softmax forward function with autograd backward hook."""
    raw_x = x.data if isinstance(x, Tensor) else x
    probs = softmax_forward_reference(raw_x, axis=axis)
    parents = (x,) if isinstance(x, Tensor) else ()
    out = Tensor(probs, parents, "dispatched_softmax")
    if not out.requires_grad:
        return out

    def _backward() -> None:
        if out.grad is None:
            return
        dx = softmax_backward(out.grad, probs, axis=axis)
        if isinstance(x, Tensor) and x.requires_grad:
            _accumulate_grad(x, dx)

    out._backward = _backward
    return out


class DispatchedSoftmax(Module):
    """Softmax activation module using dispatch autograd kernel."""

    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = int(axis)

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        return dispatched_softmax(x, axis=self.axis)

    def extra_repr(self) -> str:
        return f"axis={self.axis}"

    def __repr__(self) -> str:
        return f"DispatchedSoftmax({self.extra_repr()})"


SoftmaxModule = DispatchedSoftmax


__all__ = [
    "DispatchedSoftmax",
    "SoftmaxModule",
    "dispatched_softmax",
    "fused_masked_softmax_backward",
    "softmax_backward",
    "softmax_backward_from_logits",
    "softmax_backward_jvp",
    "softmax_backward_reference",
    "softmax_forward_reference",
]
