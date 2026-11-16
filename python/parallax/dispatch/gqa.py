"""Grouped-Query Attention (GQA) head repeat dispatch helpers."""

from __future__ import annotations

import numpy as np


def repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
    """Repeat key/value heads along axis 1 by n_rep times."""
    if n_rep <= 0:
        raise ValueError(f"n_rep must be positive, got {n_rep}")
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor (batch, heads, seq, dim), got ndim={x.ndim}")
    if n_rep == 1:
        return x

    return np.repeat(x, repeats=n_rep, axis=1)


def reduce_kv_grad(grad: np.ndarray, n_rep: int) -> np.ndarray:
    """Accumulate repeated query head gradients back to KV head dimensions."""
    if n_rep <= 0:
        raise ValueError(f"n_rep must be positive, got {n_rep}")
    if grad.ndim != 4:
        raise ValueError(f"Expected 4D grad (batch, heads, seq, dim), got ndim={grad.ndim}")
    if n_rep == 1:
        return grad
    b, total_heads, s, d = grad.shape
    if total_heads % n_rep != 0:
        raise ValueError(f"total_heads ({total_heads}) must be divisible by n_rep ({n_rep})")
    n_kv = total_heads // n_rep
    return grad.reshape(b, n_kv, n_rep, s, d).sum(axis=2)

