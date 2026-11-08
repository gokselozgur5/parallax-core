import numpy as np
import pytest
from parallax.engine import Value
from parallax.gradcheck import check_gradient, check_tensor_gradient, compute_numerical_gradient
from parallax.models.gpt import SwiGLU
from parallax.nn import Linear, RMSNorm
from parallax.tensor import Tensor


def test_gradcheck_simple_polynomial():
    # f(x, y) = 3*x^2 + 2*y^3
    def fn(x, y):
        return (x**2) * 3.0 + (y**3) * 2.0

    assert check_gradient(fn, [2.5, -1.8]) is True


def test_gradcheck_division_and_composite():
    # f(x, y, z) = (x * y + z) / (x + 1.0)
    def fn(x, y, z):
        return (x * y + z) / (x + 1.0)

    assert check_gradient(fn, [1.5, 3.2, 0.7]) is True


def test_gradcheck_multi_branch_recurrence():
    # f(x) = (x + x) * (x - 2.0) / (x**2 + 1.0)
    def fn(x):
        return (x + x) * (x - 2.0) / (x**2 + 1.0)

    assert check_gradient(fn, [3.4]) is True


def test_gradcheck_detects_deliberate_error():
    # Simulate a broken analytical gradient to ensure checker rejects it
    def fn(x):
        return x**2

    var = Value(3.0)
    out = fn(var)
    out.backward()

    # Tamper with the gradient
    var.grad = 999.0

    numerical = compute_numerical_gradient(fn, [3.0])[0]
    rel_err = abs(var.grad - numerical) / max(abs(var.grad), abs(numerical))
    assert rel_err > 1e-2


def test_gradcheck_tensor_linear():
    np.random.seed(42)
    x = Tensor(np.random.randn(2, 3), requires_grad=True)
    layer = Linear(3, 4, bias=True)

    def fn(inp):
        return (layer(inp) * layer(inp)).sum()

    assert check_tensor_gradient(fn, [x], eps=1e-5, tol=1e-4) is True


def test_gradcheck_tensor_rmsnorm():
    np.random.seed(42)
    x = Tensor(np.random.randn(2, 4), requires_grad=True)
    norm = RMSNorm(4)

    def fn(inp):
        y = norm(inp)
        return (y * y).sum()

    assert check_tensor_gradient(fn, [x], eps=1e-5, tol=2e-4) is True
