"""Tests for DispatchedLinear module."""

import numpy as np
import pytest

from parallax.backends import registry
from parallax.backends.numpy_backend import NumpyBackend
from parallax.nn import Linear, Parameter
from parallax.nn.linear import DispatchedLinear
from parallax.tensor import Tensor


@pytest.fixture(autouse=True)
def setup_backend():
    registry.clear()
    registry.register("numpy", NumpyBackend())
    yield
    registry.clear()



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


def test_dispatched_linear_repr():
    layer = DispatchedLinear(8, 4, bias=True)
    expected = "DispatchedLinear(in_features=8, out_features=4, bias=True)"
    assert repr(layer) == expected
    assert layer.extra_repr() == "in_features=8, out_features=4, bias=True"


def test_dispatched_linear_package_export():
    from parallax.nn import DispatchedLinear as DL

    assert DL is DispatchedLinear


def test_dispatched_linear_batched():
    lin = Linear(5, 2, bias=True)
    dlin = DispatchedLinear.from_linear(lin)
    x = Tensor(np.random.randn(3, 4, 5))
    out1 = lin(x)
    out2 = dlin(x)
    assert out1.shape == (3, 4, 2)
    assert np.allclose(out1.data, out2.data, atol=1e-6)


def test_dispatched_linear_backward_parity():
    lin = Linear(4, 3, bias=True)
    dlin = DispatchedLinear(4, 3, bias=True)
    dlin.weight = Parameter(lin.weight.data.copy())
    dlin.bias = Parameter(lin.bias.data.copy()) if lin.bias is not None else None
    x_data = np.random.randn(2, 4)
    x1 = Tensor(x_data.copy(), requires_grad=True)
    x2 = Tensor(x_data.copy(), requires_grad=True)
    lin(x1).sum().backward()
    dlin(x2).sum().backward()
    assert np.allclose(x1.grad, x2.grad, atol=1e-6)


def test_dispatched_linear_param_gradients():
    lin = Linear(4, 3, bias=True)
    dlin = DispatchedLinear(4, 3, bias=True)
    dlin.weight = Parameter(lin.weight.data.copy())
    dlin.bias = Parameter(lin.bias.data.copy()) if lin.bias is not None else None
    x_data = np.random.randn(2, 4)
    x1 = Tensor(x_data.copy(), requires_grad=True)
    x2 = Tensor(x_data.copy(), requires_grad=True)
    lin(x1).sum().backward()
    dlin(x2).sum().backward()
    assert np.allclose(lin.weight.grad, dlin.weight.grad, atol=1e-6)
    assert np.allclose(lin.bias.grad, dlin.bias.grad, atol=1e-6)






