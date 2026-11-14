"""Dispatched tensor operations with autograd awareness."""

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

    def _backward() -> None:
        if out.grad is None:
            return
        a_is_1d, b_is_1d = (a.ndim == 1), (b.ndim == 1)
        a_2d = a.data[None, :] if a_is_1d else a.data
        b_2d = b.data[..., :, None] if b_is_1d else b.data
        if a_is_1d and b_is_1d:
            g_2d = np.array([[float(out.grad)]])
        elif a_is_1d:
            g_2d = out.grad[..., None, :]
        elif b_is_1d:
            g_2d = out.grad[..., None]
        else:
            g_2d = out.grad

        if a.requires_grad:
            other_t = np.swapaxes(b_2d, -1, -2)
            grad_self = g_2d @ other_t
            if a_is_1d:
                grad_self = np.squeeze(grad_self, axis=-2)
            _accumulate_grad(a, grad_self)
        if b.requires_grad:
            self_t = np.swapaxes(a_2d, -1, -2)
            grad_other = self_t @ g_2d
            if b_is_1d:
                grad_other = np.squeeze(grad_other, axis=-1)
            _accumulate_grad(b, grad_other)

    out._backward = _backward
    return out
