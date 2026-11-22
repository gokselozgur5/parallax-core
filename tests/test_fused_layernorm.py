"""Regression tests for residual output backpropagation in dispatched_fused_layernorm."""

from __future__ import annotations

import numpy as np

from parallax.dispatch import dispatched_fused_layernorm
from parallax.tensor import Tensor
from python.parallax.dispatch.fused_layernorm import fused_layernorm_backward


def test_fused_layernorm_residual_output_backpropagates() -> None:
    x = Tensor(np.array([[1.0, 2.0, 4.0]]))
    res = Tensor(np.array([[0.5, -1.0, 2.0]]))
    _, fused = dispatched_fused_layernorm(x, residual=res, return_residual=True)
    np.testing.assert_allclose(fused.data, [[1.5, 1.0, 6.0]])
    fused.sum().backward()
    assert x.grad is not None and res.grad is not None, "residual stream is detached"
    np.testing.assert_allclose(x.grad, np.ones((1, 3)))
    np.testing.assert_allclose(res.grad, np.ones((1, 3)))


def test_fused_layernorm_combined_norm_and_residual_gradients() -> None:
    x_arr = np.array([[1.0, 2.0, 4.0], [-1.0, 0.5, 2.5]])
    res_arr = np.array([[0.5, -1.0, 2.0], [1.5, 0.5, -0.5]])
    w_arr = np.array([1.2, 0.8, 1.5])
    b_arr = np.array([0.1, -0.2, 0.05])
    x = Tensor(x_arr.copy())
    res = Tensor(res_arr.copy())
    w = Tensor(w_arr.copy())
    b = Tensor(b_arr.copy())
    normed, fused = dispatched_fused_layernorm(
        x, weight=w, bias=b, residual=res, eps=1e-5, return_residual=True
    )
    loss = (normed * 2.0 + fused * 3.0).sum()
    loss.backward()
    dx_norm, dw_ref, db_ref, dres_norm = fused_layernorm_backward(
        np.full_like(x_arr, 2.0), x_arr, w_arr, b_arr, residual=res_arr, eps=1e-5
    )
    assert x.grad is not None and res.grad is not None
    assert w.grad is not None and b.grad is not None
    assert dres_norm is not None
    np.testing.assert_allclose(x.grad, dx_norm + 3.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(res.grad, dres_norm + 3.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(w.grad, dw_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(b.grad, db_ref, rtol=1e-6, atol=1e-6)
