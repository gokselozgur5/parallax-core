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


def test_dropout_forward_eval_and_zero_p():
    x = np.ones((4, 4), dtype=np.float32)
    out, mask = dropout_forward_reference(x, p=0.5, training=False)
    np.testing.assert_allclose(out, x)
    assert np.all(mask)
    out_zero_p, mask_zero = dropout_forward_reference(x, p=0.0, training=True)
    np.testing.assert_allclose(out_zero_p, x)
    assert np.all(mask_zero)


def test_dropout_forward_scale_expectation():
    x = np.ones((200, 200), dtype=np.float32)
    out, mask = dropout_forward_reference(x, p=0.4, training=True, seed=123)
    np.testing.assert_allclose(np.mean(out), 1.0, atol=0.03)
