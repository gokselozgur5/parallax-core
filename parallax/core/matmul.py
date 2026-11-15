"""Autograd-aware dispatched matrix multiplication."""

from __future__ import annotations

import numpy as np
from parallax.backends.dispatch import dispatch
from parallax.tensor import Tensor, _accumulate_grad


def dispatched_matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication of two Tensors routed through dispatch('gemm')."""
    if not isinstance(a, Tensor) or not isinstance(b, Tensor):
        raise TypeError("dispatched_matmul requires Tensor arguments")

    out_data = dispatch("gemm", a.data, b.data)
    out = Tensor(out_data, (a, b), "dispatched_matmul")

    if not out.requires_grad:
        return out
    return out
