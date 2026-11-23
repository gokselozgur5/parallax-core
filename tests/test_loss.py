"""
Unit tests for Loss functions, Label Smoothing, and Perplexity metrics.
"""

import math
import numpy as np

from parallax.gradcheck import check_tensor_gradient
from parallax.loss import cross_entropy_loss, perplexity_from_loss
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

    eps, n_samples, vocab = 0.1, 2, 3
    ref_losses, ref_grads = [], []
    for row, target in zip(logits.data.tolist(), targets.tolist()):
        m = max(row)
        exps = [math.exp(v - m) for v in row]
        z = sum(exps)
        log_p = [v - m - math.log(z) for v in row]
        probs = [e / z for e in exps]
        q = [(1.0 - eps if j == target else 0.0) + eps / vocab for j in range(vocab)]
        ref_losses.append(-sum(qj * lp for qj, lp in zip(q, log_p)))
        ref_grads.append([(pj - qj) / n_samples for pj, qj in zip(probs, q)])
    assert math.isclose(
        float(loss_smooth.data), sum(ref_losses) / n_samples, rel_tol=1e-12, abs_tol=1e-12
    )
    assert np.allclose(logits_smooth.grad, np.array(ref_grads), rtol=1e-12, atol=1e-12)
    assert (
        check_tensor_gradient(
            lambda x: cross_entropy_loss(x, targets, label_smoothing=eps),
            [logits_smooth],
            eps=1e-5,
            tol=1e-5,
        )
        is True
    )


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

    ref_valid_losses, ref_valid_grads = [], []
    for row, target in zip(logits.data[:2].tolist(), targets[:2].tolist()):
        m = max(row)
        exps = [math.exp(v - m) for v in row]
        z = sum(exps)
        probs = [e / z for e in exps]
        ref_valid_losses.append(-(row[target] - m - math.log(z)))
        ref_valid_grads.append(
            [(p - (1.0 if j == target else 0.0)) / 2.0 for j, p in enumerate(probs)]
        )
    assert math.isclose(
        float(loss.data), sum(ref_valid_losses) / 2.0, rel_tol=1e-12, abs_tol=1e-12
    )
    assert np.allclose(logits.grad[:2], np.array(ref_valid_grads), rtol=1e-12, atol=1e-12)
    assert (
        check_tensor_gradient(
            lambda x: cross_entropy_loss(x, targets, ignore_index=-100),
            [logits],
            eps=1e-5,
            tol=1e-5,
        )
        is True
    )


def test_perplexity_from_loss():
    # Loss 0.0 -> PPL = 1.0
    assert math.isclose(perplexity_from_loss(0.0), 1.0, rel_tol=1e-5)

    # Loss ln(10) -> PPL = 10.0
    assert math.isclose(perplexity_from_loss(math.log(10.0)), 10.0, rel_tol=1e-5)
