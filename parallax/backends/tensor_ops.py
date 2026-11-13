"""Autograd-aware dispatched tensor operations."""

from __future__ import annotations

import numpy as np

from parallax.tensor import Tensor


def _dispatch_gemm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Route GEMM through backend dispatch with fallback."""
    try:
        from parallax.backends.dispatch import dispatch

        res = dispatch("gemm", a, b)
        return res if isinstance(res, np.ndarray) else np.asarray(res)
    except (ImportError, AttributeError):
        return a @ b


def dispatched_matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication routing forward through backend dispatch."""
    if not isinstance(a, Tensor) or not isinstance(b, Tensor):
        raise TypeError("dispatched_matmul requires Tensor arguments")
    return Tensor(_dispatch_gemm(a.data, b.data), (a, b), "@")
