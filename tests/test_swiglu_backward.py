"""Regression tests for Gated-MLP and dispatched SwiGLU Tensor autograd wiring."""

from __future__ import annotations

import numpy as np
from parallax.nn import Linear, Module
from parallax.tensor import Tensor
from python.parallax.dispatch.swiglu_backward import (
    FusedSwiGLUBackward,
    dispatched_swiglu_backward,
    gated_mlp_backward,
)


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


def test_decomposed_fused_swiglu_linear_mlp_backward() -> None:
    np.random.seed(11)
    mlp, x = _GatedMLP(3, 5, bias=True), Tensor(np.random.randn(2, 3))
    mlp(x).sum().backward()
    assert x.grad is not None
    fused_grads = [x.grad.copy()] + [p.grad.copy() for p in mlp.parameters() if p.grad is not None]
    mlp.zero_grad()
    x2 = Tensor(x.data.copy())
    mlp.w_down(FusedSwiGLUBackward()(mlp.w_gate(x2), mlp.w_up(x2))).sum().backward()
    decomp_grads = [x2.grad] + [p.grad for p in mlp.parameters()]
    for gf, gd in zip(fused_grads, decomp_grads):
        assert gd is not None
        np.testing.assert_allclose(gd, gf, rtol=1e-6, atol=1e-7)


def test_gated_mlp_backward_populates_parameter_grads_and_const_mode() -> None:
    np.random.seed(19)
    mlp, x = _GatedMLP(3, 4, bias=False), Tensor(np.random.randn(2, 3))
    gate, up = x.data @ mlp.w_gate.weight.data.T, x.data @ mlp.w_up.weight.data.T
    dx, dwg, dwu, dwd = gated_mlp_backward(
        np.ones((2, 3)), x, gate, up, mlp.w_gate.weight, mlp.w_up.weight, mlp.w_down.weight
    )
    assert x.grad is not None and mlp.w_gate.weight.grad is not None
    assert mlp.w_up.weight.grad is not None and mlp.w_down.weight.grad is not None
    np.testing.assert_allclose(x.grad, dx)
    np.testing.assert_allclose(mlp.w_gate.weight.grad, dwg)
    np.testing.assert_allclose(mlp.w_up.weight.grad, dwu)
    np.testing.assert_allclose(mlp.w_down.weight.grad, dwd)
    const_out = FusedSwiGLUBackward.gated_mlp(
        x.data, mlp.w_gate.weight.data, mlp.w_up.weight.data, mlp.w_down.weight.data
    )
    assert const_out.requires_grad is False
