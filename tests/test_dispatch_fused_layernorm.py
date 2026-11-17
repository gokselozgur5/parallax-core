"""Tests for Dispatched LayerNorm with RMSNorm weight scaling and bias fusion."""

from __future__ import annotations

import numpy as np

from parallax.tensor import Tensor
from python.parallax.dispatch.fused_layernorm import (
    DispatchedFusedLayerNorm,
    FusedLayerNormRMS,
    dispatched_fused_layernorm,
    fused_layernorm_backward,
    fused_layernorm_forward,
    rms_weight_scale,
    rms_weight_scale_backward,
    stabilize_epsilon,
)


def test_stabilize_epsilon():
    assert stabilize_epsilon(1e-5) == 1e-5
    assert stabilize_epsilon(0.0) == 1e-12
    assert stabilize_epsilon(-2.0) == 1e-12


def test_rms_weight_scale_and_backward():
    np.random.seed(42)
    w = np.random.randn(8).astype(np.float64)
    dout = np.random.randn(8).astype(np.float64)
    scaled = rms_weight_scale(w)
    expected_scale = w / np.sqrt(np.mean(w ** 2) + 1e-6)
    np.testing.assert_allclose(scaled, expected_scale, rtol=1e-5)
    ana_dw = rms_weight_scale_backward(dout, w)
    h = 1e-6
    num_dw = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy(); w_p[i] += h
        w_m = w.copy(); w_m[i] -= h
        num_dw[i] = np.sum((rms_weight_scale(w_p) - rms_weight_scale(w_m)) / (2 * h) * dout)
    np.testing.assert_allclose(ana_dw, num_dw, rtol=1e-5, atol=1e-6)


def test_fused_layernorm_forward_shapes_and_parity():
    np.random.seed(99)
    x = np.random.randn(2, 4, 16).astype(np.float32)
    w = np.ones(16, dtype=np.float32)
    b = np.zeros(16, dtype=np.float32)
    out = fused_layernorm_forward(x, w, b, eps=1e-5)
    assert out.shape == (2, 4, 16)
    mean = np.mean(out, axis=-1)
    std = np.std(out, axis=-1)
    np.testing.assert_allclose(mean, np.zeros((2, 4)), atol=1e-5)
    np.testing.assert_allclose(std, np.ones((2, 4)), atol=1e-3)


def test_fused_layernorm_residual_and_bias():
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    res = np.array([[0.5, 0.5], [1.0, 1.0]], dtype=np.float32)
    b = np.array([0.1, -0.1], dtype=np.float32)
    out, out_res = fused_layernorm_forward(x, bias=b, residual=res, return_residual=True)
    np.testing.assert_allclose(out_res, x + res)
    fused = x + res
    mean = np.mean(fused, axis=-1, keepdims=True)
    var = np.var(fused, axis=-1, keepdims=True)
    expected = (fused - mean) / np.sqrt(var + 1e-5) + b
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_fused_layernorm_rms_weight_scale():
    x = np.array([[2.0, 4.0], [1.0, 3.0]], dtype=np.float32)
    w = np.array([2.0, 2.0], dtype=np.float32)
    out_std = fused_layernorm_forward(x, weight=w, rms_scale=False)
    out_rms = fused_layernorm_forward(x, weight=w, rms_scale=True)
    expected_w = w / np.sqrt(np.mean(w ** 2) + 1e-5)
    np.testing.assert_allclose(expected_w, np.array([1.0, 1.0], dtype=np.float32), rtol=1e-3)
    np.testing.assert_allclose(out_rms * 2.0, out_std, rtol=1e-3)
