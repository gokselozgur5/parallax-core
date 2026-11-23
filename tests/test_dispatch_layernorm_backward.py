"""Tests for LayerNorm backward autograd functional kernel."""

from __future__ import annotations

import numpy as np

from parallax.dispatch.layernorm_backward import (
    stabilize_epsilon,
)


def test_stabilize_epsilon():
    assert stabilize_epsilon(1e-5) == 1e-5
    assert stabilize_epsilon(0.0) == 1e-12
    assert stabilize_epsilon(-5.0) == 1e-12


from parallax.dispatch.layernorm_backward import (
    layernorm_backward_reference,
)


def test_layernorm_backward_shapes_and_values():
    np.random.seed(42)
    x = np.random.randn(2, 4, 8).astype(np.float32)
    w = np.ones(8, dtype=np.float32)
    b = np.zeros(8, dtype=np.float32)
    dout = np.ones_like(x)
    dx, dw, db = layernorm_backward_reference(dout, x, w, b, eps=1e-5)
    assert dx.shape == x.shape
    assert dw.shape == w.shape
    assert db.shape == b.shape
    assert not np.isnan(dx).any()
    assert not np.isnan(dw).any()
    assert not np.isnan(db).any()


from parallax.dispatch.layernorm_backward import (
    layernorm_backward,
)


def test_layernorm_backward_matches_reference():
    np.random.seed(123)
    x = np.random.randn(3, 16).astype(np.float32)
    w = np.random.randn(16).astype(np.float32)
    b = np.random.randn(16).astype(np.float32)
    dout = np.random.randn(3, 16).astype(np.float32)
    dx_ref, dw_ref, db_ref = layernorm_backward_reference(dout, x, w, b, eps=1e-5)
    dx, dw, db = layernorm_backward(dout, x, w, b, eps=1e-5)
    np.testing.assert_allclose(dx, dx_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dw, dw_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(db, db_ref, rtol=1e-6, atol=1e-6)


def test_layernorm_backward_numerical_finite_difference():
    np.random.seed(99)
    x = np.random.randn(2, 4).astype(np.float64)
    w = np.random.randn(4).astype(np.float64)
    b = np.random.randn(4).astype(np.float64)
    dout = np.random.randn(2, 4).astype(np.float64)
    eps = 1e-5
    dx, dw, db = layernorm_backward(dout, x, w, b, eps=eps)
    h = 1e-6
    def f(xi, wi, bi):
        m, v = np.mean(xi, axis=-1, keepdims=True), np.var(xi, axis=-1, keepdims=True)
        return ((xi - m) / np.sqrt(v + eps)) * wi + bi
    num_dx = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            xp, xm = x.copy(), x.copy()
            xp[i, j] += h; xm[i, j] -= h
            num_dx[i, j] = np.sum((f(xp, w, b) - f(xm, w, b)) / (2 * h) * dout)
    np.testing.assert_allclose(dx, num_dx, rtol=1e-4, atol=1e-5)


from parallax.dispatch.layernorm_backward import (
    fused_layernorm_backward,
)


def test_fused_layernorm_backward():
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    res = np.array([[0.5, -0.5], [1.0, -1.0]], dtype=np.float32)
    w = np.ones(2, dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    dout = np.ones_like(x)
    dx, dw, db, dres = fused_layernorm_backward(dout, x, w, b, residual=res, eps=1e-5)
    dx_d, dw_d, db_d = layernorm_backward(dout, x + res, w, b, eps=1e-5)
    np.testing.assert_allclose(dx, dx_d)
    np.testing.assert_allclose(dw, dw_d)
    np.testing.assert_allclose(db, db_d)
    np.testing.assert_allclose(dres, dx_d)


from parallax.tensor import Tensor
from parallax.dispatch.layernorm_backward import (
    dispatched_layernorm,
)


def test_dispatched_layernorm_autograd():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
    w = Tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
    b = Tensor(np.array([0.5, -0.5], dtype=np.float32), requires_grad=True)
    out = dispatched_layernorm(x, w, b, eps=1e-5)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert w.grad is not None
    assert b.grad is not None
    assert x.grad.shape == (2, 2)
    assert w.grad.shape == (2,)
    assert b.grad.shape == (2,)


from parallax.dispatch.layernorm_backward import (
    DispatchedLayerNormBackward,
)


def test_dispatched_layernorm_backward_class():
    op = DispatchedLayerNormBackward(eps=1e-5)
    x = np.random.randn(2, 4).astype(np.float32)
    w = np.ones(4, dtype=np.float32)
    b = np.zeros(4, dtype=np.float32)
    dout = np.ones_like(x)
    dx, dw, db = op(dout, x, w, b)
    assert dx.shape == x.shape
    assert dw.shape == w.shape
    assert db.shape == b.shape


def test_layernorm_backward_exports():
    import sys
    lb = sys.modules["parallax.dispatch.layernorm_backward"]
    for sym in (
        "DispatchedLayerNormBackward",
        "dispatched_layernorm",
        "fused_layernorm_backward",
        "layernorm_backward",
        "layernorm_backward_reference",
        "stabilize_epsilon",
    ):
        assert hasattr(lb, sym)


def test_layernorm_backward_package_exports():
    import parallax.dispatch as pd
    for sym in (
        "DispatchedLayerNormBackward",
        "layernorm_backward",
        "layernorm_backward_reference",
    ):
        assert hasattr(pd, sym)
