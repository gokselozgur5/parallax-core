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
