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
