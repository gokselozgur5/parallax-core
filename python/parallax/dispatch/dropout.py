"""Dispatched Dropout functional forward and backward mask generator."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from parallax.nn import Module
from parallax.tensor import Tensor, _accumulate_grad


def dropout_mask_generator(
    shape: Tuple[int, ...],
    p: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate boolean dropout keep mask with probability (1 - p)."""
    if p <= 0.0:
        return np.ones(shape, dtype=bool)
    if p >= 1.0:
        return np.zeros(shape, dtype=bool)
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=shape) >= p
