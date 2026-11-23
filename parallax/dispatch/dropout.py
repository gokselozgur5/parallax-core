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


def dropout_forward(
    x: np.ndarray,
    p: float = 0.5,
    training: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute functional forward pass for Dropout layer."""
    return dropout_forward_reference(x, p=p, training=training, seed=seed)


def dropout_backward(
    dout: np.ndarray,
    mask: np.ndarray,
    p: float = 0.5,
    training: bool = True,
) -> np.ndarray:
    """Execute functional backward pass for Dropout layer."""
    return dropout_backward_reference(dout, mask, p=p, training=training)


def dispatched_dropout(
    x: Tensor | np.ndarray,
    p: float = 0.5,
    training: bool = True,
    seed: Optional[int] = None,
) -> Tensor:
    """Dispatched Dropout forward function with autograd backward hook."""
    raw_x = x.data if isinstance(x, Tensor) else x
    out_arr, mask = dropout_forward(raw_x, p=p, training=training, seed=seed)
    parents = (x,) if isinstance(x, Tensor) else ()
    out = Tensor(out_arr, parents, "dispatched_dropout")
    if not out.requires_grad:
        return out

    def _backward() -> None:
        if out.grad is None:
            return
        dx = dropout_backward(out.grad, mask, p=p, training=training)
        if isinstance(x, Tensor) and x.requires_grad:
            _accumulate_grad(x, dx)

    out._backward = _backward
    return out


class DispatchedDropout(Module):
    """Dropout regularization module using dispatch autograd kernel."""

    def __init__(self, p: float = 0.5, training: bool = True, seed: Optional[int] = None) -> None:
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Dropout probability p must be in [0, 1], got {p}")
        self.p = float(p)
        self.training = bool(training)
        self.seed = seed

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        return dispatched_dropout(x, p=self.p, training=self.training, seed=self.seed)

    def train(self, mode: bool = True) -> DispatchedDropout:
        self.training = bool(mode)
        return self

    def eval(self) -> DispatchedDropout:
        self.training = False
        return self

    def __repr__(self) -> str:
        return f"DispatchedDropout(p={self.p}, training={self.training})"


DropoutModule = DispatchedDropout

__all__ = [
    "DispatchedDropout",
    "DropoutModule",
    "dispatched_dropout",
    "dropout_backward",
    "dropout_backward_reference",
    "dropout_forward",
    "dropout_forward_reference",
    "dropout_mask_generator",
]
