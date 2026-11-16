"""Grouped-Query Attention (GQA) head repeat dispatch helpers."""

from __future__ import annotations

import numpy as np

from parallax.nn import Module
from parallax.tensor import Tensor, _accumulate_grad




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


def dispatched_repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Repeat key/value Tensor heads along axis 1 with autograd tracking."""
    out_data = repeat_kv(x.data, n_rep)
    out = Tensor(out_data, (x,), f"RepeatKV(n_rep={n_rep})")

    def _backward() -> None:
        if out.grad is None:
            return
        if x.requires_grad:
            dx = reduce_kv_grad(out.grad, n_rep)
            _accumulate_grad(x, dx)

    out._backward = _backward
    return out


class GQAHeadRepeater(Module):
    """Module repeating key/value heads for Grouped-Query Attention."""

    def __init__(self, n_rep: int) -> None:
        super().__init__()
        if n_rep <= 0:
            raise ValueError(f"n_rep must be positive, got {n_rep}")
        self.n_rep = n_rep

    def forward(self, x: Tensor) -> Tensor:
        return dispatched_repeat_kv(x, self.n_rep)

    def extra_repr(self) -> str:
        return f"n_rep={self.n_rep}"

    def __repr__(self) -> str:
        return f"GQAHeadRepeater({self.extra_repr()})"


def repeat_kv_pair(k: Tensor, v: Tensor, n_rep: int) -> tuple[Tensor, Tensor]:
    """Repeat both key and value heads along axis 1 by n_rep times."""
    return dispatched_repeat_kv(k, n_rep), dispatched_repeat_kv(v, n_rep)


__all__ = [
    "GQAHeadRepeater",
    "dispatched_repeat_kv",
    "reduce_kv_grad",
    "repeat_kv",
    "repeat_kv_pair",
]





