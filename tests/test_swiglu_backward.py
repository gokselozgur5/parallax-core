"""Regression tests for Gated-MLP and dispatched SwiGLU Tensor autograd wiring."""

from __future__ import annotations

import numpy as np
from parallax.nn import Linear, Module
from parallax.tensor import Tensor
from python.parallax.dispatch.swiglu_backward import FusedSwiGLUBackward, dispatched_swiglu_backward


class _GatedMLP(Module):
    def __init__(self, in_dim: int, hidden_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.w_gate = Linear(in_dim, hidden_dim, bias=bias)
        self.w_up = Linear(in_dim, hidden_dim, bias=bias)
        self.w_down = Linear(hidden_dim, in_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return FusedSwiGLUBackward.gated_mlp(x, self.w_gate, self.w_up, self.w_down)


def test_dispatched_swiglu_backward_propagates_to_linear_weights() -> None:
    np.random.seed(0)
    mlp, x = _GatedMLP(3, 4, bias=True), Tensor(np.random.randn(2, 3))
    gate, up = mlp.w_gate(x), mlp.w_up(x)
    dg, du = dispatched_swiglu_backward(Tensor(np.ones((2, 4))), gate, up)
    assert gate in dg._prev and up in du._prev
    assert mlp.w_gate.weight.grad is not None and mlp.w_up.weight.grad is not None
    assert x.grad is not None


def test_gated_mlp_autograd_loss_backward_finite_diff() -> None:
    np.random.seed(7)
    mlp = _GatedMLP(3, 4, bias=True)
    x, target = Tensor(np.random.randn(2, 3)), np.random.randn(2, 3)
    (mlp(x) * Tensor(target, requires_grad=False)).sum().backward()
    for arr, grad in [(x.data, x.grad)] + [(p.data, p.grad) for p in mlp.parameters()]:
        assert grad is not None
        num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            orig = arr[idx]
            arr[idx] = orig + 1e-6
            pos = float((mlp(Tensor(x.data, requires_grad=False)).data * target).sum())
            arr[idx] = orig - 1e-6
            neg = float((mlp(Tensor(x.data, requires_grad=False)).data * target).sum())
            arr[idx], num[idx] = orig, (pos - neg) / 2e-6
        np.testing.assert_allclose(grad, num, rtol=1e-5, atol=1e-6)
