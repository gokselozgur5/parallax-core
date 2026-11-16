"""Tests for Logit soft-capping / tanh-capping dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from python.parallax.dispatch.logit_cap import (
    DispatchedLogitCap,
    logit_soft_cap,
    logit_soft_cap_backward,
)



def test_logit_soft_cap_forward_values() -> None:
    x = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
    cap = 30.0
    out = logit_soft_cap(x, cap=cap)
    expected = cap * np.tanh(x / cap)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_logit_soft_cap_invalid_cap_raises() -> None:
    x = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="cap must be strictly positive"):
        logit_soft_cap(x, cap=0.0)
    with pytest.raises(ValueError, match="cap must be strictly positive"):
        logit_soft_cap(x, cap=-5.0)


def test_logit_soft_cap_extreme_bounds_and_dtype() -> None:
    x = np.array([-1e6, 0.0, 1e6], dtype=np.float32)
    out = logit_soft_cap(x, cap=20.0)
    assert out.dtype == np.float32
    assert np.isclose(out[0], -20.0, atol=1e-5)
    assert np.isclose(out[1], 0.0, atol=1e-5)
    assert np.isclose(out[2], 20.0, atol=1e-5)


def test_logit_soft_cap_backward_values() -> None:
    x = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
    grad_out = np.ones_like(x)
    cap = 30.0
    grad_x = logit_soft_cap_backward(grad_out, x, cap=cap)
    t = np.tanh(x / cap)
    expected = grad_out * (1.0 - t * t)
    np.testing.assert_allclose(grad_x, expected, rtol=1e-5, atol=1e-6)


def test_logit_soft_cap_finite_difference_gradient() -> None:
    x = np.array([-2.5, 0.5, 3.2], dtype=np.float64)
    grad_out = np.array([0.8, -1.2, 0.5], dtype=np.float64)
    cap = 15.0
    analytic_grad = logit_soft_cap_backward(grad_out, x, cap=cap)
    eps = 1e-6
    num_grad = np.zeros_like(x)
    for i in range(len(x)):
        x_pos, x_neg = x.copy(), x.copy()
        x_pos[i] += eps
        x_neg[i] -= eps
        diff = (logit_soft_cap(x_pos, cap=cap) - logit_soft_cap(x_neg, cap=cap)) * grad_out
        num_grad[i] = np.sum(diff) / (2 * eps)
    np.testing.assert_allclose(analytic_grad, num_grad, rtol=1e-5, atol=1e-5)


def test_dispatched_logit_cap_module_init_and_forward() -> None:
    mod = DispatchedLogitCap(cap=20.0)
    assert mod.cap == 20.0
    x = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
    out = mod(x)
    expected = 20.0 * np.tanh(x / 20.0)
    np.testing.assert_allclose(out.data, expected, rtol=1e-5, atol=1e-6)





