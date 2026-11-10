"""
Unit tests for BitNet b1.58 ternary quantization (M30).
"""

import numpy as np
import pytest
from parallax.bitnet import (
    BitLinear,
    quantize_activations_int8,
    quantize_weights_ternary,
)
from parallax.tensor import Tensor


def test_quantize_weights_ternary_strictly_ternary():
    np.random.seed(42)
    w = np.random.randn(32, 64)
    w_ternary, gamma = quantize_weights_ternary(w)

    # Values must be strictly in {-1, 0, 1}
    unique_vals = set(np.unique(w_ternary))
    assert unique_vals.issubset({-1.0, 0.0, 1.0})
    assert gamma > 0.0


def test_quantize_activations_int8_range():
    x = np.random.randn(2, 8, 32) * 5.0
    x_quant, scale = quantize_activations_int8(x)

    assert x_quant.shape == x.shape
    assert np.all(x_quant >= -128.0)
    assert np.all(x_quant <= 127.0)
    assert np.all(scale > 0.0)


def test_bitlinear_forward_and_ste_backward():
    B, T, in_f, out_f = 2, 4, 16, 8
    layer = BitLinear(in_features=in_f, out_features=out_f, bias=True)

    x = Tensor(np.random.randn(B, T, in_f), requires_grad=True)
    out = layer(x)

    assert out.shape == (B, T, out_f)
    assert np.all(np.isfinite(out.data))

    loss = (out * out).sum()
    loss.backward()

    # Straight-Through Estimator must update continuous weights
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == (out_f, in_f)
    assert x.grad is not None
    assert x.grad.shape == (B, T, in_f)


def test_bitlinear_ste_backward_analytical_gamma_scaling():
    layer = BitLinear(in_features=4, out_features=2, bias=False)
    x = Tensor(np.random.randn(1, 3, 4), requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    w_ternary, gamma = quantize_weights_ternary(layer.weight.data)
    expected_x_grad = np.matmul(np.ones_like(out.data), w_ternary) * gamma
    assert np.allclose(x.grad, expected_x_grad, atol=1e-7)


def test_bitlinear_forward_against_independent_reference():
    """Verify BitLinear forward against explicit independent scalar reference."""
    np.random.seed(42)
    B, T, in_f, out_f = 2, 3, 8, 4
    layer = BitLinear(in_f, out_f, bias=True)
    layer.bias.data = np.linspace(0.5, 2.0, out_f)
    x = Tensor(np.random.randn(B, T, in_f))
    out = layer(x)

    w = layer.weight.data
    gamma = float(np.mean(np.abs(w))) + 1e-12
    w_ternary = np.clip(np.round(w / gamma), -1.0, 1.0)
    x_scale = 127.0 / (np.max(np.abs(x.data), axis=-1, keepdims=True) + 1e-12)
    x_quant = np.clip(np.round(x.data * x_scale), -128.0, 127.0)
    expected = (x_quant @ w_ternary.T) * (gamma / x_scale) + layer.bias.data
    np.testing.assert_allclose(out.data, expected, rtol=1e-5, atol=1e-5)


def test_bitlinear_forward_gamma_scaling_sensitivity():
    """Verify forward pass scales proportionally with gamma; kills gamma-dropped fault."""
    layer = BitLinear(in_features=4, out_features=2, bias=False)
    x = Tensor(np.array([[[1.0, 2.0, -1.0, -2.0]]]))
    out1 = layer(x).data.copy()

    # Scale weights by 5.0: w_ternary remains identical, gamma increases by 5.0x
    layer.weight.data = layer.weight.data * 5.0
    out2 = layer(x).data.copy()

    np.testing.assert_allclose(out2, out1 * 5.0, rtol=1e-5, atol=1e-5)
    assert not np.allclose(out2, out1, atol=1e-3)


def test_bitlinear_nonzero_bias_forward_and_backward():
    """Verify non-zero bias affects forward output and receives backward gradients."""
    layer = BitLinear(in_features=4, out_features=3, bias=True)
    bias_vec = np.array([1.5, -2.5, 4.0])
    layer.bias.data = bias_vec.copy()
    x = Tensor(np.random.randn(2, 5, 4), requires_grad=True)
    out = layer(x)

    # Shift test against zero-bias layer
    layer_nobias = BitLinear(in_features=4, out_features=3, bias=False)
    layer_nobias.weight.data = layer.weight.data.copy()
    out_nobias = layer_nobias(x)
    expected_shift = np.broadcast_to(bias_vec, out.shape)
    np.testing.assert_allclose(out.data - out_nobias.data, expected_shift, atol=1e-6)

    loss = (out * 2.0).sum()
    loss.backward()
    expected_bias_grad = np.sum(2.0 * np.ones_like(out.data).reshape(-1, 3), axis=0)
    np.testing.assert_allclose(layer.bias.grad, expected_bias_grad, atol=1e-7)


def test_bitlinear_ste_weight_grad_finite_diff():
    np.random.seed(42)
    B, T, in_f, out_f = 2, 3, 4, 5
    layer = BitLinear(in_features=in_f, out_features=out_f, bias=False)
    x_data = np.random.randn(B, T, in_f)
    x = Tensor(x_data, requires_grad=False)
    out = layer(x)
    v = np.random.randn(*out.shape)
    (out * Tensor(v, requires_grad=False)).sum().backward()

    x_quant, x_scale = quantize_activations_int8(x_data)
    x_dequant = x_quant / x_scale
    eps = 1e-6
    grad_num = np.zeros_like(layer.weight.data)
    for i in range(out_f):
        for j in range(in_f):
            w_p, w_m = layer.weight.data.copy(), layer.weight.data.copy()
            w_p[i, j] += eps
            w_m[i, j] -= eps
            lp = np.sum(v * (x_dequant @ w_p.T))
            lm = np.sum(v * (x_dequant @ w_m.T))
            grad_num[i, j] = (lp - lm) / (2 * eps)

    assert np.allclose(layer.weight.grad, grad_num, atol=1e-6)




