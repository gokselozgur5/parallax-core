"""Unit tests for SwiGLU activation layer and dispatch helpers."""

import numpy as np
import pytest
from python.parallax.dispatch.swiglu import swiglu_dispatch


def test_swiglu_dispatch_parity() -> None:
    a = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = swiglu_dispatch(a, b)
    sig = 1.0 / (1.0 + np.exp(-a))
    expected = a * sig * b
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_swiglu_dispatch_validation() -> None:
    a = np.ones((2, 3), dtype=np.float32)
    b = np.ones((2, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        swiglu_dispatch(a, b)
    with pytest.raises(TypeError):
        swiglu_dispatch(np.array([1 + 2j]), np.array([2 + 3j]))


def test_swiglu_backward_dispatch() -> None:
    from python.parallax.dispatch.swiglu import swiglu_backward_dispatch

    a = np.array([1.0, -1.0], dtype=np.float32)
    b = np.array([2.0, 3.0], dtype=np.float32)
    grad = np.array([1.0, 1.0], dtype=np.float32)
    ga, gb = swiglu_backward_dispatch(grad, a, b)
    sig = 1.0 / (1.0 + np.exp(-a))
    silu = a * sig
    dsilu = sig * (1.0 + a * (1.0 - sig))
    np.testing.assert_allclose(gb, grad * silu, atol=1e-5)
    np.testing.assert_allclose(ga, grad * b * dsilu, atol=1e-5)


def test_dispatched_swiglu_tensor_autograd() -> None:
    from parallax.tensor import Tensor
    from python.parallax.dispatch.swiglu import dispatched_swiglu

    g_data = np.array([[1.0, -2.0], [0.5, 1.5]], dtype=np.float32)
    u_data = np.array([[2.0, 3.0], [-1.0, 0.5]], dtype=np.float32)
    g = Tensor(g_data, requires_grad=True)
    u = Tensor(u_data, requires_grad=True)
    out = dispatched_swiglu(g, u)
    assert out.shape == (2, 2)
    out.sum().backward()
    assert g.grad is not None
    assert u.grad is not None



