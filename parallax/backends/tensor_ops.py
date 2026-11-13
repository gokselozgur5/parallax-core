"""Autograd-aware dispatched tensor operations."""

from __future__ import annotations

import numpy as np

from parallax.tensor import Tensor, _accumulate_grad


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
    out = Tensor(_dispatch_gemm(a.data, b.data), (a, b), "@")
    if not out.requires_grad:
        return out

    def _backward() -> None:
        if out.grad is None:
            return
        a_1d, b_1d = (a.ndim == 1), (b.ndim == 1)
        a_2d = a.data[None, :] if a_1d else a.data
        b_2d = b.data[..., :, None] if b_1d else b.data
        g = np.array([[float(out.grad)]]) if (a_1d and b_1d) else (
            out.grad[..., None, :] if a_1d else (out.grad[..., None] if b_1d else out.grad)
        )
        if a.requires_grad:
            ga = g @ np.swapaxes(b_2d, -1, -2)
            _accumulate_grad(a, np.squeeze(ga, -2) if a_1d else ga)
        if b.requires_grad:
            gb = np.swapaxes(a_2d, -1, -2) @ g
            _accumulate_grad(b, np.squeeze(gb, -1) if b_1d else gb)

    out._backward = _backward
    return out
