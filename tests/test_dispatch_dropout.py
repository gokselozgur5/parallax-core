"""Tests for Dispatched Dropout functional forward and backward mask generator."""

from __future__ import annotations

import numpy as np

from parallax.tensor import Tensor
from python.parallax.dispatch.dropout import (
    DispatchedDropout,
    DropoutModule,
    dispatched_dropout,
    dropout_backward,
    dropout_backward_reference,
    dropout_forward,
    dropout_forward_reference,
    dropout_mask_generator,
)


def test_dropout_mask_generator_shapes_and_prob():
    shape = (100, 100)
    mask = dropout_mask_generator(shape, p=0.3, seed=42)
    assert mask.shape == shape
    assert mask.dtype == bool
    keep_ratio = np.mean(mask)
    np.testing.assert_allclose(keep_ratio, 0.7, atol=0.03)
