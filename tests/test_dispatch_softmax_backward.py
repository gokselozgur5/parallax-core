"""Tests for Dispatched Softmax backward Jacobian vector product helper."""

from __future__ import annotations

import numpy as np

from parallax.tensor import Tensor
from python.parallax.dispatch.softmax_backward import (
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
