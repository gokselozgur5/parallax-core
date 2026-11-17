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


def dropout_forward_reference(
    x: np.ndarray,
    p: float = 0.5,
    training: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute reference inverted dropout forward pass."""
    if not training or p <= 0.0:
        mask = np.ones(x.shape, dtype=bool)
        return x.copy().astype(x.dtype), mask
    if p >= 1.0:
        mask = np.zeros(x.shape, dtype=bool)
        return np.zeros_like(x), mask
    mask = dropout_mask_generator(x.shape, p=p, seed=seed)
    scale = 1.0 / (1.0 - p)
    out = (x * mask * scale).astype(x.dtype)
    return out, mask


def dropout_backward_reference(
    dout: np.ndarray,
    mask: np.ndarray,
    p: float = 0.5,
    training: bool = True,
) -> np.ndarray:
    """Execute reference inverted dropout backward gradient pass."""
    if not training or p <= 0.0:
        return dout.copy().astype(dout.dtype)
    if p >= 1.0:
        return np.zeros_like(dout)
    scale = 1.0 / (1.0 - p)
    dx = (dout * mask * scale).astype(dout.dtype)
    return dx
