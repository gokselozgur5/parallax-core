"""Tests for Dispatched Fast Cross-Entropy loss kernel and module."""

from __future__ import annotations

import numpy as np
import pytest

from parallax.tensor import Tensor
from python.parallax.dispatch.cross_entropy import (
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
