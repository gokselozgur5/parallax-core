"""Tests for fused RMSNorm forward and backward residual kernels."""

from __future__ import annotations

import numpy as np

from parallax.tensor import Tensor
from python.parallax.dispatch.fused_rmsnorm_residual import (
    FusedRMSNormResidual,
    dispatched_fused_rmsnorm_residual,
    fused_rmsnorm_residual_backward,
    fused_rmsnorm_residual_backward_reference,
    fused_rmsnorm_residual_forward,
    fused_rmsnorm_residual_forward_reference,
    stabilize_epsilon,
)


def test_stabilize_epsilon():
    assert stabilize_epsilon(1e-5) == 1e-5
    assert stabilize_epsilon(0.0) == 1e-12
    assert stabilize_epsilon(-5.0) == 1e-12


def test_fused_rmsnorm_residual_forward_shapes():
    np.random.seed(42)
    x = np.random.randn(2, 4, 8).astype(np.float32)
    res = np.random.randn(2, 4, 8).astype(np.float32)
    w = np.ones(8, dtype=np.float32)
    normed, fused = fused_rmsnorm_residual_forward_reference(x, res, w, eps=1e-5)
    assert normed.shape == x.shape
    assert fused.shape == res.shape
    np.testing.assert_allclose(fused, x + res)
    assert not np.isnan(normed).any()
    assert not np.isnan(fused).any()


def test_fused_rmsnorm_residual_forward_dispatch():
    np.random.seed(123)
    x = np.random.randn(3, 16).astype(np.float32)
    res = np.random.randn(3, 16).astype(np.float32)
    w = np.random.randn(16).astype(np.float32)
    normed_ref, fused_ref = fused_rmsnorm_residual_forward_reference(x, res, w, eps=1e-5)
    normed, fused = fused_rmsnorm_residual_forward(x, res, w, eps=1e-5)
    np.testing.assert_allclose(normed, normed_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(fused, fused_ref, rtol=1e-6, atol=1e-6)


def test_fused_rmsnorm_residual_backward_shapes():
    np.random.seed(77)
    x = np.random.randn(2, 4, 8).astype(np.float32)
    res = np.random.randn(2, 4, 8).astype(np.float32)
    w = np.ones(8, dtype=np.float32)
    dout = np.ones_like(x)
    fused = x + res
    dx, dw, dres = fused_rmsnorm_residual_backward_reference(dout, fused, w, eps=1e-5)
    assert dx.shape == x.shape
    assert dw.shape == w.shape
    assert dres.shape == res.shape
    assert not np.isnan(dx).any()
    assert not np.isnan(dw).any()
    assert not np.isnan(dres).any()


def test_fused_rmsnorm_residual_backward_dispatch():
    np.random.seed(88)
    x = np.random.randn(3, 16).astype(np.float32)
    res = np.random.randn(3, 16).astype(np.float32)
    w = np.random.randn(16).astype(np.float32)
    dout = np.random.randn(3, 16).astype(np.float32)
    fused = x + res
    dx_ref, dw_ref, dres_ref = fused_rmsnorm_residual_backward_reference(dout, fused, w, eps=1e-5)
    dx, dw, dres = fused_rmsnorm_residual_backward(dout, fused, w, eps=1e-5)
    np.testing.assert_allclose(dx, dx_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dw, dw_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dres, dres_ref, rtol=1e-6, atol=1e-6)


def test_fused_rmsnorm_residual_backward_numerical():
    np.random.seed(99)
    x = np.random.randn(2, 4).astype(np.float64)
    res = np.random.randn(2, 4).astype(np.float64)
    w = np.random.randn(4).astype(np.float64)
    dout = np.random.randn(2, 4).astype(np.float64)
    eps = 1e-5
    fused = x + res
    dx, dw, _ = fused_rmsnorm_residual_backward(dout, fused, w, eps=eps)
    h = 1e-6

    def f(xi, wi):
        rms = np.sqrt(np.mean(xi ** 2, axis=-1, keepdims=True) + eps)
        return (xi / rms) * wi

    num_dx = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            fp = fused.copy(); fp[i, j] += h
            fn = fused.copy(); fn[i, j] -= h
            num_dx[i, j] = np.sum((f(fp, w) - f(fn, w)) / (2 * h) * dout)
    np.testing.assert_allclose(dx, num_dx, rtol=1e-4, atol=1e-5)


def test_fused_rmsnorm_residual_incoming_dres():
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    res = np.array([[0.5, -0.5], [1.0, -1.0]], dtype=np.float32)
    w = np.ones(2, dtype=np.float32)
    dout = np.ones_like(x)
    dres_in = np.full_like(x, 2.0)
    fused = x + res
    dx_base, dw_base, _ = fused_rmsnorm_residual_backward(dout, fused, w, eps=1e-5)
    dx, dw, dres = fused_rmsnorm_residual_backward(
        dout, fused, w, eps=1e-5, dresidual=dres_in
    )
    np.testing.assert_allclose(dw, dw_base)
    np.testing.assert_allclose(dx, dx_base + dres_in)
    np.testing.assert_allclose(dres, dx_base + dres_in)
