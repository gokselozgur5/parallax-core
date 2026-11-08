"""
Unit tests for Selective State Space Model (Mamba/S6) (M28).
"""

import numpy as np
import pytest
from parallax.mamba import discretize_zoh, selective_scan_forward


def test_discretize_zoh_shapes_and_exponential_decay():
    B, T, D, N = 2, 4, 8, 16
    delta = np.full((B, T, D), 0.1)
    # Diagonal transition matrix A (negative for stability)
    A = -np.ones((D, N))
    B_in = np.ones((B, T, N))

    A_bar, B_bar = discretize_zoh(delta, A, B_in)
    assert A_bar.shape == (B, T, D, N)
    assert B_bar.shape == (B, T, D, N)
    # exp(0.1 * -1) ~ 0.904837
    np.testing.assert_allclose(A_bar[0, 0, 0, 0], np.exp(-0.1), atol=1e-6)
    # 0.1 * 1.0 = 0.1
    np.testing.assert_allclose(B_bar[0, 0, 0, 0], 0.1, atol=1e-6)


def test_selective_scan_forward_recurrence_and_continuity():
    B, T, D, N = 1, 3, 2, 4
    u = np.ones((B, T, D))
    delta = np.full((B, T, D), 0.5)
    A = -np.ones((D, N))
    B_param = np.ones((B, T, N))
    C_param = np.ones((B, T, N))
    D_skip = np.array([0.5, 0.5])

    y, h_states = selective_scan_forward(u, delta, A, B_param, C_param, D=D_skip)

    assert y.shape == (B, T, D)
    assert h_states.shape == (B, T, D, N)
    # Output must strictly increase as state accumulates input over time
    assert y[0, 1, 0] > y[0, 0, 0]
    assert y[0, 2, 0] > y[0, 1, 0]

