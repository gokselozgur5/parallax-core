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
