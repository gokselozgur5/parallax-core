"""Tests for Dispatched Fast Cross-Entropy loss kernel and module."""

from __future__ import annotations

import numpy as np
import pytest

from parallax.tensor import Tensor
from parallax.dispatch.cross_entropy import (
    CrossEntropyLossModule,
    DispatchedCrossEntropyLoss,
    cross_entropy_backward_reference,
    cross_entropy_forward_reference,
    fast_cross_entropy,
    online_logsumexp,
)


def test_online_logsumexp_numerical_stability():
    x = np.array([[1000.0, 1001.0], [-1000.0, -999.0]], dtype=np.float32)
    lse, m, s = online_logsumexp(x, axis=-1)
    assert not np.isnan(lse).any()
    assert not np.isinf(lse).any()


def test_online_logsumexp_chunked_parity():
    np.random.seed(42)
    x = np.random.randn(4, 128).astype(np.float32)
    lse_full, _, _ = online_logsumexp(x, chunk_size=None, axis=-1)
    lse_chunked, _, _ = online_logsumexp(x, chunk_size=16, axis=-1)
    np.testing.assert_allclose(lse_full, lse_chunked, rtol=1e-5, atol=1e-5)


def test_target_validation_out_of_bounds():
    logits = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        cross_entropy_forward_reference(logits, np.array([-1, 0]))
    with pytest.raises(ValueError):
        cross_entropy_forward_reference(logits, np.array([0, 4]))


def test_cross_entropy_forward_reductions():
    logits = np.array([[2.0, 1.0], [0.5, 2.5]], dtype=np.float32)
    targets = np.array([0, 1], dtype=np.int64)
    loss_mean, _, _ = cross_entropy_forward_reference(logits, targets, reduction="mean")
    loss_sum, _, _ = cross_entropy_forward_reference(logits, targets, reduction="sum")
    loss_none, _, _ = cross_entropy_forward_reference(logits, targets, reduction="none")
    assert loss_none.shape == (2,)
    np.testing.assert_allclose(loss_mean, np.mean(loss_none), rtol=1e-6)
    np.testing.assert_allclose(loss_sum, np.sum(loss_none), rtol=1e-6)


def test_cross_entropy_label_smoothing_and_ignore_index():
    logits = np.array([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]], dtype=np.float32)
    targets = np.array([0, -100], dtype=np.int64)
    loss_raw, _, valid = cross_entropy_forward_reference(logits, targets, ignore_index=-100)
    loss_smooth, _, _ = cross_entropy_forward_reference(
        logits, targets, ignore_index=-100, label_smoothing=0.1
    )
    assert valid[0] and not valid[1]
    assert float(loss_smooth) > float(loss_raw)


def test_cross_entropy_backward_shapes_and_masking():
    logits = np.array([[2.0, 1.0, -1.0], [0.0, 3.0, 1.0]], dtype=np.float32)
    targets = np.array([0, -100], dtype=np.int64)
    _, lse, _ = cross_entropy_forward_reference(logits, targets, ignore_index=-100)
    grad = cross_entropy_backward_reference(1.0, logits, targets, lse, ignore_index=-100)
    assert grad.shape == logits.shape
    np.testing.assert_allclose(grad[1], np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(np.sum(grad[0]), 0.0, atol=1e-6)


def test_cross_entropy_backward_finite_difference():
    np.random.seed(99)
    x = np.random.randn(2, 4).astype(np.float64)
    targets = np.array([1, 2], dtype=np.int64)
    _, lse, _ = cross_entropy_forward_reference(x, targets, label_smoothing=0.1)
    grad = cross_entropy_backward_reference(1.0, x, targets, lse, label_smoothing=0.1)
    h = 1e-6
    num_grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_pos = x.copy(); x_pos[i, j] += h
            x_neg = x.copy(); x_neg[i, j] -= h
            lp, _, _ = cross_entropy_forward_reference(x_pos, targets, label_smoothing=0.1)
            ln, _, _ = cross_entropy_forward_reference(x_neg, targets, label_smoothing=0.1)
            num_grad[i, j] = (lp - ln) / (2 * h)
    np.testing.assert_allclose(grad, num_grad, rtol=1e-5, atol=1e-6)


def test_fast_cross_entropy_autograd_and_module():
    x = Tensor(np.array([[2.0, 1.0], [0.5, 2.5]], dtype=np.float32), requires_grad=True)
    targets = np.array([0, 1], dtype=np.int64)
    loss = fast_cross_entropy(x, targets)
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (2, 2)
    assert not np.isnan(x.grad).any()
    mod = DispatchedCrossEntropyLoss(label_smoothing=0.05)
    assert isinstance(mod, CrossEntropyLossModule)
    out = mod(x, targets)
    assert out.shape == ()


def test_fast_cross_entropy_accepts_tensor_targets():
    logits = np.array([[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    targets = np.array([0, 1])
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    expected = -log_probs[[0, 1], targets].mean()
    out = fast_cross_entropy(Tensor(logits), Tensor(targets, requires_grad=False))
    np.testing.assert_allclose(out.data, expected, rtol=1e-12)


def test_cross_entropy_exports():
    import parallax.dispatch.cross_entropy as ce
    import parallax.dispatch as pd
    expected = (
        "CrossEntropyLossModule",
        "DispatchedCrossEntropyLoss",
        "cross_entropy_backward_reference",
        "cross_entropy_forward_reference",
        "fast_cross_entropy",
        "online_logsumexp",
    )
    for sym in expected:
        assert hasattr(ce, sym)
        assert hasattr(pd, sym)
