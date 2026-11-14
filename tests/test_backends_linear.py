import pytest
from parallax.backends import registry
from parallax.backends.numpy_backend import NumpyBackend
import numpy as np
from parallax.backends.linear import DispatchedLinear
from parallax.nn import Linear, Parameter
from parallax.tensor import Tensor


@pytest.fixture(autouse=True)
def setup_backend():
    registry.clear()
    registry.register("numpy", NumpyBackend())
    yield
    registry.clear()


def test_dispatched_linear_forward_parity() -> None:
    for bias in [True, False]:
        lin = Linear(4, 3, bias=bias)
        dlin = DispatchedLinear.from_linear(lin)
        x_data = np.random.randn(2, 4)
        x1 = Tensor(x_data.copy())
        x2 = Tensor(x_data.copy())
        out1 = lin(x1)
        out2 = dlin(x2)
        assert out1.shape == (2, 3)
        assert np.allclose(out1.data, out2.data, atol=1e-6)


def test_dispatched_linear_batched() -> None:
    lin = Linear(5, 2, bias=True)
    dlin = DispatchedLinear.from_linear(lin)
    x = Tensor(np.random.randn(3, 4, 5))
    out1 = lin(x)
    out2 = dlin(x)
    assert out1.shape == (3, 4, 2)
    assert np.allclose(out1.data, out2.data, atol=1e-6)


def test_dispatched_linear_backward_parity() -> None:
    lin = Linear(4, 3, bias=True)
    dlin = DispatchedLinear(4, 3, bias=True)
    dlin.weight = Parameter(lin.weight.data.copy())
    dlin.bias = Parameter(lin.bias.data.copy()) if lin.bias is not None else None
    x_data = np.random.randn(2, 4)
    x1, x2 = Tensor(x_data.copy(), requires_grad=True), Tensor(x_data.copy(), requires_grad=True)
    lin(x1).sum().backward()
    dlin(x2).sum().backward()
    assert np.allclose(x1.grad, x2.grad, atol=1e-6)
    assert np.allclose(lin.weight.grad, dlin.weight.grad, atol=1e-6)


def test_dispatched_linear_param_identity() -> None:
    lin = Linear(3, 2, bias=True)
    dlin = DispatchedLinear.from_linear(lin)
    assert dlin.weight is lin.weight
    assert dlin.bias is lin.bias
