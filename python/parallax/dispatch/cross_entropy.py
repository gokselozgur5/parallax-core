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
    shape[axis] = 1
    m = np.full(shape, -np.inf, dtype=x.dtype)
    s = np.zeros(shape, dtype=x.dtype)
    for start in range(0, dim, chunk_size):
        end = min(start + chunk_size, dim)
        chunk = np.take(x, indices=range(start, end), axis=axis)
        chunk_m = np.max(chunk, axis=axis, keepdims=True)
        new_m = np.maximum(m, chunk_m)
        s = s * np.exp(m - new_m) + np.sum(
            np.exp(chunk - new_m), axis=axis, keepdims=True
        )
        m = new_m
    lse = m + np.log(s)
    return lse, m, s
