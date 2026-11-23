"""Regression tests for epsilon stabilization in FusedLayerNorm backward."""

import numpy as np

from parallax.tensor import Tensor
from python.parallax.dispatch.layernorm import FusedLayerNorm


def test_fused_layernorm_zero_eps_constant_row_finite_grad() -> None:
    norm = FusedLayerNorm(8, eps=0.0)
    x = Tensor(np.ones((1, 8), dtype=np.float32), requires_grad=True)
    dout = np.arange(8, dtype=np.float32).reshape(1, 8)
    out = norm(x)
    out.backward(dout)
    assert x.grad is not None and norm.weight is not None and norm.weight.grad is not None
    assert np.all(np.isfinite(x.grad))
    np.testing.assert_allclose(x.grad, 1e6 * (dout - dout.mean()), rtol=1e-5)
    np.testing.assert_allclose(norm.weight.grad, np.zeros(8, dtype=np.float32))


def test_fused_layernorm_tiny_eps_backward_matches_stabilized_eps() -> None:
    norm = FusedLayerNorm(4, eps=0.0, elementwise_affine=False)
    x_np = np.array([[0.0, 1e-8, -1e-8, 0.0]], dtype=np.float64)
    dout = np.array([[1.0, -2.0, 0.5, 0.5]], dtype=np.float64)
    x = Tensor(x_np, requires_grad=True)
    norm(x).backward(dout)
    inv_std = 1.0 / np.sqrt(np.var(x_np, axis=-1, keepdims=True) + 1e-12)
    x_hat = (x_np - np.mean(x_np, axis=-1, keepdims=True)) * inv_std
    expected = inv_std * (dout - dout.mean(axis=-1, keepdims=True) - x_hat * np.mean(dout * x_hat))
    np.testing.assert_allclose(x.grad, expected, rtol=1e-9, atol=1e-9)
