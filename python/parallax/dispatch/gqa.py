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
