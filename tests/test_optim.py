"""
Unit tests for Gradient Clipping and Learning Rate Schedules.
"""

import math
import numpy as np

from parallax.nn import Parameter
from parallax.optim import AdamW, CosineAnnealingWithWarmup, clip_grad_norm_


def test_clip_grad_norm_scales_gradients():
    p1 = Parameter(np.array([3.0, 4.0]))
    p1.grad = np.array([3.0, 4.0])  # norm = 5.0

    p2 = Parameter(np.array([12.0]))
    p2.grad = np.array([12.0])  # total norm = sqrt(5^2 + 12^2) = 13.0

    total_norm = clip_grad_norm_([p1, p2], max_norm=6.5)
    assert math.isclose(total_norm, 13.0, rel_tol=1e-5)

    # After clipping to 6.5 (half of 13.0), each gradient should be scaled by 0.5
    assert np.allclose(p1.grad, np.array([1.5, 2.0]))
    assert np.allclose(p2.grad, np.array([6.0]))

    # If norm is smaller than max_norm, gradients should be unchanged
    p3 = Parameter(np.array([1.0]))
    p3.grad = np.array([2.0])
    norm = clip_grad_norm_([p3], max_norm=10.0)
    assert norm == 2.0
    assert p3.grad[0] == 2.0


def test_cosine_annealing_with_warmup():
    p = Parameter(np.array([1.0]))
    opt = AdamW([p], lr=0.1)

    scheduler = CosineAnnealingWithWarmup(
        opt,
        warmup_steps=10,
        total_steps=100,
        min_lr=0.01,
        base_lr=0.1,
    )

    # Step 0: Initial
    assert scheduler.get_lr(0) == 0.01

    # Step 5: Halfway warmup
    assert math.isclose(scheduler.get_lr(5), 0.055, rel_tol=1e-5)

    # Step 10: Peak LR
    assert math.isclose(scheduler.get_lr(10), 0.1, rel_tol=1e-5)

    # Step 55: Midpoint cosine decay (cos(pi/2) = 0 -> 0.01 + 0.09 * 0.5 = 0.055)
    assert math.isclose(scheduler.get_lr(55), 0.055, rel_tol=1e-5)

    # Step 100: Min LR
    assert math.isclose(scheduler.get_lr(100), 0.01, rel_tol=1e-5)

    # Beyond total steps
    assert scheduler.get_lr(150) == 0.01

    # Test scheduler.step() updates optimizer.lr
    new_lr = scheduler.step()
    assert opt.lr == new_lr
