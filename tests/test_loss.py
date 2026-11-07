"""
Unit tests for Loss functions, Label Smoothing, and Perplexity metrics.
"""

import math
import numpy as np
import pytest

from parallax.loss import compute_perplexity, cross_entropy_loss
from parallax.tensor import Tensor


def test_cross_entropy_label_smoothing():
    logits = Tensor(np.array([[2.0, 1.0, 0.1], [0.5, 3.0, 1.2]]))
    targets = np.array([0, 1])

    # Unsmoothed loss
    loss_raw = cross_entropy_loss(logits, targets, label_smoothing=0.0)

    # Smoothed loss (regularized)
    logits_smooth = Tensor(np.array([[2.0, 1.0, 0.1], [0.5, 3.0, 1.2]]))
    loss_smooth = cross_entropy_loss(logits_smooth, targets, label_smoothing=0.1)

    assert loss_smooth.data > loss_raw.data
    loss_smooth.backward()
    assert logits_smooth.grad is not None
    assert np.allclose(np.sum(logits_smooth.grad, axis=-1), [0.0, 0.0], atol=1e-5)


def test_cross_entropy_ignore_index():
    # 3 samples, 3 classes, last sample is padding (ignore_index = -100)
    logits = Tensor(np.array([[2.0, 1.0, 0.1], [0.5, 3.0, 1.2], [10.0, -10.0, 0.0]]))
    targets = np.array([0, 1, -100])

    loss = cross_entropy_loss(logits, targets, ignore_index=-100)
    loss.backward()

    # The ignored sample must have zero gradient
    assert np.allclose(logits.grad[2], [0.0, 0.0, 0.0])
    # The valid samples must have nonzero gradients
    assert not np.allclose(logits.grad[0], [0.0, 0.0, 0.0])


def test_compute_perplexity():
    # Loss 0.0 -> PPL = 1.0
    assert math.isclose(compute_perplexity(0.0), 1.0, rel_tol=1e-5)

    # Loss ln(10) -> PPL = 10.0
    assert math.isclose(compute_perplexity(math.log(10.0)), 10.0, rel_tol=1e-5)
