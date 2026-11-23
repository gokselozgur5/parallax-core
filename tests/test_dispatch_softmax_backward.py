"""Tests for Dispatched Softmax backward Jacobian vector product helper."""

from __future__ import annotations

import numpy as np

from parallax.tensor import Tensor
from parallax.dispatch.softmax_backward import (
    DispatchedSoftmax,
    SoftmaxModule,
    dispatched_softmax,
    fused_masked_softmax_backward,
    softmax_backward,
    softmax_backward_from_logits,
    softmax_backward_jvp,
    softmax_backward_reference,
    softmax_forward_reference,
)


def test_softmax_forward_numerical_stability():
    x = np.array([[1000.0, 1001.0], [-1000.0, -999.0]], dtype=np.float32)
    probs = softmax_forward_reference(x, axis=-1)
    assert not np.isnan(probs).any()
    np.testing.assert_allclose(np.sum(probs, axis=-1), np.ones(2), rtol=1e-5)


def test_softmax_backward_shapes_and_values():
    x = np.array([[1.0, 2.0, 3.0], [0.5, -0.5, 1.0]], dtype=np.float32)
    probs = softmax_forward_reference(x)
    dout = np.ones_like(probs)
    dx = softmax_backward_reference(dout, probs)
    assert dx.shape == x.shape
    np.testing.assert_allclose(np.sum(dx, axis=-1), np.zeros(2), atol=1e-6)


def test_softmax_backward_matches_reference():
    np.random.seed(42)
    x = np.random.randn(4, 8).astype(np.float32)
    dout = np.random.randn(4, 8).astype(np.float32)
    probs = softmax_forward_reference(x)
    dx_ref = softmax_backward_reference(dout, probs)
    dx = softmax_backward(dout, probs)
    np.testing.assert_allclose(dx, dx_ref, rtol=1e-6, atol=1e-6)


def test_softmax_backward_finite_difference():
    np.random.seed(99)
    x = np.random.randn(2, 4).astype(np.float64)
    dout = np.random.randn(2, 4).astype(np.float64)
    probs, dx = softmax_backward_from_logits(dout, x)
    h = 1e-6
    num_dx = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_pos = x.copy(); x_pos[i, j] += h
            x_neg = x.copy(); x_neg[i, j] -= h
            f_pos = softmax_forward_reference(x_pos)
            f_neg = softmax_forward_reference(x_neg)
            num_dx[i, j] = np.sum((f_pos - f_neg) / (2 * h) * dout)
    np.testing.assert_allclose(dx, num_dx, rtol=1e-5, atol=1e-6)


def test_softmax_backward_jvp():
    probs = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    v = np.array([1.0, -1.0, 2.0], dtype=np.float64)
    jvp = softmax_backward_jvp(probs, v)
    diag_p = np.diag(probs)
    outer_p = np.outer(probs, probs)
    jacobian = diag_p - outer_p
    expected = jacobian @ v
    np.testing.assert_allclose(jvp, expected, rtol=1e-6, atol=1e-6)


def test_fused_masked_softmax_backward():
    probs = np.array([[0.5, 0.5], [0.3, 0.7]], dtype=np.float32)
    dout = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask = np.array([[False, True], [False, False]], dtype=bool)
    dx_masked = fused_masked_softmax_backward(dout, probs, mask=mask)
    dout_eff = np.array([[1.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    dx_ref = softmax_backward_reference(dout_eff, probs)
    np.testing.assert_allclose(dx_masked, dx_ref)


def test_dispatched_softmax_autograd():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
    out = dispatched_softmax(x, axis=-1)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (2, 2)
    assert not np.isnan(x.grad).any()


def test_dispatched_softmax_module():
    mod = DispatchedSoftmax(axis=-1)
    assert isinstance(mod, SoftmaxModule)
    x = Tensor(np.array([[0.0, 1.0]], dtype=np.float32), requires_grad=True)
    out = mod(x)
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out.data.sum(axis=-1), np.ones(1), rtol=1e-5)


def test_softmax_backward_exports():
    import importlib
    sb = importlib.import_module("parallax.dispatch.softmax_backward")
    for sym in (
        "DispatchedSoftmax",
        "SoftmaxModule",
        "dispatched_softmax",
        "fused_masked_softmax_backward",
        "softmax_backward",
        "softmax_backward_from_logits",
        "softmax_backward_jvp",
        "softmax_backward_reference",
        "softmax_forward_reference",
    ):
        assert hasattr(sb, sym)


def test_softmax_backward_package_exports():
    import parallax.dispatch as pd
    for sym in (
        "DispatchedSoftmax",
        "SoftmaxModule",
        "dispatched_softmax",
        "fused_masked_softmax_backward",
        "softmax_backward",
        "softmax_backward_from_logits",
        "softmax_backward_jvp",
        "softmax_backward_reference",
        "softmax_forward_reference",
    ):
        assert hasattr(pd, sym)
