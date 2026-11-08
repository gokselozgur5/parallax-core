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


