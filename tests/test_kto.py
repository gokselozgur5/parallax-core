"""
Unit tests for Kahneman-Tversky Optimization (KTO) (M29).
"""

import numpy as np
import pytest
from parallax.kto import kto_value_function


def test_kto_value_function_asymmetry_and_derivatives():
    z = np.array([0.5, -0.5])
    # For desirable, positive z should yield low loss
    loss_d, grad_d = kto_value_function(z, is_desirable=True, lambda_d=1.0)
    assert loss_d[0] < loss_d[1]
    # Gradient of loss w.r.t z must be negative (increasing z reduces loss)
    assert np.all(grad_d < 0.0)

    # For undesirable, positive z should yield high loss
    loss_u, grad_u = kto_value_function(z, is_desirable=False, lambda_u=1.33)
    assert loss_u[0] > loss_u[1]
    # Gradient of loss w.r.t z must be positive (increasing z increases loss)
    assert np.all(grad_u > 0.0)
