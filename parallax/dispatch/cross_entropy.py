"""Dispatched Fast Cross-Entropy loss with online log-sum-exp stabilization."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from parallax.nn import Module
from parallax.tensor import Tensor, _accumulate_grad


def online_logsumexp(
    x: np.ndarray,
    chunk_size: Optional[int] = None,
    axis: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute log-sum-exp stably using online streaming max and sum-exp."""
    if chunk_size is None or chunk_size >= x.shape[axis]:
        m = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - m)
        s = np.sum(exp_x, axis=axis, keepdims=True)
        lse = m + np.log(s)
        return lse, m, s
    dim = x.shape[axis]
    shape = list(x.shape)
