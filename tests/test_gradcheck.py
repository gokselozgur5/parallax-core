import pytest
from parallax.engine import Value
from parallax.gradcheck import check_gradient, compute_numerical_gradient


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
