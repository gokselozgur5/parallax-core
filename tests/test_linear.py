"""Tests for DispatchedLinear module."""

import numpy as np
import pytest

from parallax.nn import Linear
from parallax.nn.linear import DispatchedLinear
from parallax.tensor import Tensor


def test_dispatched_linear_initialization():
    layer = DispatchedLinear(4, 3, bias=True)
    assert layer.in_features == 4
    assert layer.out_features == 3
    assert layer.weight.shape == (3, 4)
    assert layer.bias is not None
    assert layer.bias.shape == (3,)


def test_dispatched_linear_forward_parity():
    for bias in [True, False]:
        lin = Linear(4, 3, bias=bias)
        dlin = DispatchedLinear.from_linear(lin)
        x_data = np.random.randn(2, 4)
        x1, x2 = Tensor(x_data.copy()), Tensor(x_data.copy())
        out1 = lin(x1)
        out2 = dlin(x2)
        assert out1.shape == (2, 3)
        assert np.allclose(out1.data, out2.data, atol=1e-6)

