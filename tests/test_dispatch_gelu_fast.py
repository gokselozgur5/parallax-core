"""Tests for GeLU fast approximation and tanh forward dispatch."""

from __future__ import annotations

import math

import numpy as np
import pytest

from parallax.dispatch.gelu_fast import (
    DispatchedGELUFast,
    FastGELU,
    GELUFast,
    dispatched_gelu_fast,
    gelu_fast,
    gelu_fast_backward,
    gelu_fast_dispatch,
    gelu_fast_forward_np,
)
from parallax.tensor import Tensor


def test_gelu_fast_forward_parity() -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal((4, 16), dtype=np.float32)
    out_np = gelu_fast_forward_np(x)
    out_disp = gelu_fast_dispatch(x)
    assert np.allclose(out_np, out_disp, atol=1e-6)


def test_gelu_fast_bounds_and_edges() -> None:
    x = np.array([-20.0, -10.0, 0.0, 10.0, 20.0], dtype=np.float32)
    out = gelu_fast_dispatch(x)
    assert np.isclose(out[0], 0.0)
    assert np.isclose(out[2], 0.0)
    assert np.isclose(out[4], 20.0)

    empty = np.empty((0, 8), dtype=np.float32)
    assert gelu_fast_dispatch(empty).shape == (0, 8)

    with pytest.raises(TypeError, match="Complex dtypes"):
        gelu_fast_dispatch(np.array([1 + 2j], dtype=np.complex64))


def test_gelu_fast_backward_numerical() -> None:
    rng = np.random.default_rng(123)
    x = rng.uniform(-4.0, 4.0, size=(3, 5)).astype(np.float32)
    grad_out = rng.standard_normal((3, 5), dtype=np.float32)
    analytical = gelu_fast_backward(grad_out, x)

    eps = 1e-3
    numerical = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        x_pos = x.copy()
        x_neg = x.copy()
        x_pos[idx] += eps
        x_neg[idx] -= eps
        f_pos = gelu_fast_forward_np(x_pos)
        f_neg = gelu_fast_forward_np(x_neg)
        numerical[idx] = np.sum(grad_out * (f_pos - f_neg) / (2.0 * eps))

    assert np.allclose(analytical, numerical, rtol=1e-2, atol=1e-3)


def test_dispatched_gelu_fast_autograd() -> None:
    x_np = np.array([-1.5, 0.0, 1.5], dtype=np.float32)
    t = Tensor(x_np.copy(), requires_grad=True)
    out = dispatched_gelu_fast(t)
    loss = (out * 2.0).sum()
    loss.backward()
    assert t.grad is not None
    assert t.grad.shape == t.shape

    k = math.sqrt(2.0 / math.pi)
    x64 = x_np.astype(np.float64)
    th = np.tanh(k * (x64 + 0.044715 * x64**3))
    dgelu = 0.5 * (1.0 + th) + 0.5 * x64 * (1.0 - th**2) * k * (1.0 + 3.0 * 0.044715 * x64**2)
    expected = (2.0 * dgelu).astype(np.float32)
    assert np.allclose(t.grad, expected, rtol=1e-5, atol=1e-6)
    assert np.allclose(t.grad, gelu_fast_backward(np.full(3, 2.0, dtype=np.float32), t.data), rtol=1e-6, atol=1e-6)

    eps = 1e-3
    num_grad = np.zeros_like(x_np)
    for i in range(x_np.size):
        xp, xm = x_np.copy(), x_np.copy()
        xp[i] += eps
        xm[i] -= eps
        fp = float((dispatched_gelu_fast(Tensor(xp)) * 2.0).sum().data)
        fm = float((dispatched_gelu_fast(Tensor(xm)) * 2.0).sum().data)
        num_grad[i] = (fp - fm) / (2.0 * eps)
    assert np.allclose(t.grad, num_grad, rtol=1e-2, atol=1e-3)

    w = np.array([-1.25, 0.5, 3.0], dtype=np.float32)
    t2 = Tensor(x_np.copy(), requires_grad=True)
    (dispatched_gelu_fast(t2) * Tensor(w)).sum().backward()
    assert t2.grad is not None
    assert np.allclose(t2.grad, w * dgelu.astype(np.float32), rtol=1e-5, atol=1e-6)


def test_dispatched_gelu_fast_module() -> None:
    mod = DispatchedGELUFast()
    x = Tensor(np.array([[0.5, -0.5]], dtype=np.float32))
    out = mod(x)
    assert out.shape == (1, 2)
    assert repr(mod) == "DispatchedGELUFast(approximate='tanh')"
    assert FastGELU is DispatchedGELUFast
    assert GELUFast is DispatchedGELUFast
    assert gelu_fast is gelu_fast_dispatch
    with pytest.raises(ValueError, match="Unsupported approximation"):
        DispatchedGELUFast(approximate="invalid")
