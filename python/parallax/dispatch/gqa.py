"""Grouped-Query Attention (GQA) head repeat dispatch helpers."""

from __future__ import annotations

import numpy as np


def repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
    """Repeat key/value heads along axis 1 by n_rep times."""
    if n_rep == 1:
        return x
    return np.repeat(x, repeats=n_rep, axis=1)
