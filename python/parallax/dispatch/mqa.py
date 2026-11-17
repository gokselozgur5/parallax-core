"""Multi-Query Attention (MQA) head broadcast and KV expansion helpers."""

from __future__ import annotations

import numpy as np

from parallax.nn import Module
from parallax.tensor import Tensor, _accumulate_grad


def broadcast_kv(x: np.ndarray, num_heads: int) -> np.ndarray:
    """Broadcast single KV head across num_heads query heads."""
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if x.ndim == 3:
        x = np.expand_dims(x, axis=1)
    elif x.ndim != 4:
        raise ValueError(f"Expected 3D or 4D tensor, got ndim={x.ndim}")
    if x.shape[1] != 1:
        raise ValueError(f"MQA requires single KV head, got shape[1]={x.shape[1]}")
    return np.repeat(x, repeats=num_heads, axis=1)


def reduce_mqa_grad(grad: np.ndarray, original_ndim: int = 4) -> np.ndarray:
    """Accumulate broadcast query head gradients back to KV head dimensions."""
    if grad.ndim != 4:
        raise ValueError(f"Expected 4D grad, got ndim={grad.ndim}")
    summed = grad.sum(axis=1, keepdims=True)
    if original_ndim == 3:
        return np.squeeze(summed, axis=1)
    if original_ndim == 4:
        return summed
    raise ValueError(f"Expected original_ndim 3 or 4, got {original_ndim}")


def dispatched_broadcast_kv(x: Tensor, num_heads: int) -> Tensor:
    """Broadcast KV Tensor to num_heads with autograd tracking."""
    orig_ndim = x.data.ndim
    out_data = broadcast_kv(x.data, num_heads)
    out = Tensor(out_data, (x,), f"BroadcastKV(num_heads={num_heads})")

    def _backward() -> None:
        if out.grad is None:
            return
        if x.requires_grad:
            dx = reduce_mqa_grad(out.grad, original_ndim=orig_ndim)
            _accumulate_grad(x, dx)

    out._backward = _backward
    return out


def broadcast_mqa_pair(k: Tensor, v: Tensor, num_heads: int) -> tuple[Tensor, Tensor]:
    """Broadcast both key and value heads to num_heads."""
    return dispatched_broadcast_kv(k, num_heads), dispatched_broadcast_kv(v, num_heads)


class MQAHeadBroadcaster(Module):
    """Module broadcasting key/value head for Multi-Query Attention."""

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        self.num_heads = num_heads

    def forward(self, x: Tensor) -> Tensor:
        return dispatched_broadcast_kv(x, self.num_heads)

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}"

    def __repr__(self) -> str:
        return f"MQAHeadBroadcaster({self.extra_repr()})"


__all__ = [
    "MQAHeadBroadcaster",
    "broadcast_kv",
    "broadcast_mqa_pair",
    "dispatched_broadcast_kv",
    "reduce_mqa_grad",
]
