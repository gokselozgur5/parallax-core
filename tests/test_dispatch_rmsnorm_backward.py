"""Tests for RMSNorm backward autograd functional kernel."""

from __future__ import annotations

import numpy as np
import pytest

from parallax.tensor import Tensor
from python.parallax.dispatch.rmsnorm_backward import (
    rmsnorm_backward,
    rmsnorm_backward_reference,
    stabilize_epsilon,
)


def test_stabilize_epsilon():
    assert stabilize_epsilon(1e-5) == 1e-5
    assert stabilize_epsilon(0.0) == 1e-12
    assert stabilize_epsilon(-5.0) == 1e-12


def test_rmsnorm_backward_shapes_and_values():
    np.random.seed(42)
    x = np.random.randn(2, 4, 8).astype(np.float32)
    w = np.ones(8, dtype=np.float32)
    dout = np.ones_like(x)
    dx, dw = rmsnorm_backward_reference(dout, x, w, eps=1e-5)
    assert dx.shape == x.shape
    assert dw.shape == w.shape
    assert not np.isnan(dx).any()
    assert not np.isnan(dw).any()


def test_rmsnorm_backward_matches_reference():
    np.random.seed(123)
    x = np.random.randn(3, 16).astype(np.float32)
    w = np.random.randn(16).astype(np.float32)
    dout = np.random.randn(3, 16).astype(np.float32)
    dx_ref, dw_ref = rmsnorm_backward_reference(dout, x, w, eps=1e-5)
    dx, dw = rmsnorm_backward(dout, x, w, eps=1e-5)
    np.testing.assert_allclose(dx, dx_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dw, dw_ref, rtol=1e-6, atol=1e-6)


def test_rmsnorm_backward_numerical_finite_difference():
    np.random.seed(99)
    x = np.random.randn(2, 4).astype(np.float64)
    w = np.random.randn(4).astype(np.float64)
    dout = np.random.randn(2, 4).astype(np.float64)
    eps = 1e-5
    dx, dw = rmsnorm_backward(dout, x, w, eps=eps)
    h = 1e-6

    def f(xi, wi):
        rms = np.sqrt(np.mean(xi ** 2, axis=-1, keepdims=True) + eps)
        return (xi / rms) * wi

    num_dx = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_pos = x.copy(); x_pos[i, j] += h
            x_neg = x.copy(); x_neg[i, j] -= h
            num_dx[i, j] = np.sum((f(x_pos, w) - f(x_neg, w)) / (2 * h) * dout)
    np.testing.assert_allclose(dx, num_dx, rtol=1e-4, atol=1e-5)
