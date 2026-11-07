import pytest
from parallax.engine import Value
from parallax.gradcheck import check_gradient


def test_relu_forward_and_backward():
    a = Value(2.0)
    b = a.relu()
    b.backward()
    assert b.data == 2.0
    assert a.grad == 1.0

    c = Value(-3.0)
    d = c.relu()
    d.backward()
    assert d.data == 0.0
    assert c.grad == 0.0


def test_tanh_forward_and_gradcheck():
    def fn(x):
        return x.tanh()

    assert check_gradient(fn, [0.5], eps=1e-5, tol=1e-4) is True
    assert check_gradient(fn, [-1.2], eps=1e-5, tol=1e-4) is True


def test_sigmoid_forward_and_gradcheck():
    def fn(x):
        return x.sigmoid()

    assert check_gradient(fn, [0.8], eps=1e-5, tol=1e-4) is True
    assert check_gradient(fn, [-2.5], eps=1e-5, tol=1e-4) is True


def test_gelu_forward_and_gradcheck():
    def fn(x):
        return x.gelu()

    # Test across multiple critical points: negative, zero, positive
    assert check_gradient(fn, [1.5], eps=1e-5, tol=1e-4) is True
    assert check_gradient(fn, [-0.75], eps=1e-5, tol=1e-4) is True
    assert check_gradient(fn, [0.0], eps=1e-5, tol=1e-4) is True


def test_exp_and_log_gradcheck():
    def fn(x, y):
        return x.exp() + y.log()

    assert check_gradient(fn, [1.2, 3.4], eps=1e-5, tol=1e-4) is True
