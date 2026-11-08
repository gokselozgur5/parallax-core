"""
Unit tests for Selective State Space Model (Mamba/S6) (M28).
"""

import numpy as np
import pytest
from parallax.mamba import MambaBlock, discretize_zoh, selective_scan_backward, selective_scan_forward
from parallax.tensor import Tensor


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


def test_selective_scan_backward_finite_difference():
    np.random.seed(42)
    B, T, D, N = 1, 3, 2, 2
    u = np.random.randn(B, T, D)
    delta = np.random.uniform(0.1, 0.5, size=(B, T, D))
    A = -np.random.uniform(0.5, 2.0, size=(D, N))
    B_param = np.random.randn(B, T, N)
    C_param = np.random.randn(B, T, N)
    D_skip = np.random.randn(D)

    y, h_states = selective_scan_forward(u, delta, A, B_param, C_param, D=D_skip)
    d_y = np.random.randn(*y.shape)

    du, ddelta, dA, dB, dC, dD = selective_scan_backward(
        d_y, u, delta, A, B_param, C_param, D=D_skip, h_states=h_states
    )

    eps = 1e-6
    # Check du
    num_du = np.zeros_like(u)
    for idx in np.ndindex(u.shape):
        orig = u[idx]
        u[idx] = orig + eps
        y_pos, _ = selective_scan_forward(u, delta, A, B_param, C_param, D=D_skip)
        u[idx] = orig - eps
        y_neg, _ = selective_scan_forward(u, delta, A, B_param, C_param, D=D_skip)
        u[idx] = orig
        num_du[idx] = np.sum(d_y * (y_pos - y_neg)) / (2.0 * eps)
    np.testing.assert_allclose(du, num_du, atol=1e-5)

    # Check ddelta
    num_ddelta = np.zeros_like(delta)
    for idx in np.ndindex(delta.shape):
        orig = delta[idx]
        delta[idx] = orig + eps
        y_pos, _ = selective_scan_forward(u, delta, A, B_param, C_param, D=D_skip)
        delta[idx] = orig - eps
        y_neg, _ = selective_scan_forward(u, delta, A, B_param, C_param, D=D_skip)
        delta[idx] = orig
        num_ddelta[idx] = np.sum(d_y * (y_pos - y_neg)) / (2.0 * eps)
    np.testing.assert_allclose(ddelta, num_ddelta, atol=1e-5)

    # Check dA
    num_dA = np.zeros_like(A)
    for idx in np.ndindex(A.shape):
        orig = A[idx]
        A[idx] = orig + eps
        y_pos, _ = selective_scan_forward(u, delta, A, B_param, C_param, D=D_skip)
        A[idx] = orig - eps
        y_neg, _ = selective_scan_forward(u, delta, A, B_param, C_param, D=D_skip)
        A[idx] = orig
        num_dA[idx] = np.sum(d_y * (y_pos - y_neg)) / (2.0 * eps)
    np.testing.assert_allclose(dA, num_dA, atol=1e-5)

    # Check dB
    num_dB = np.zeros_like(B_param)
    for idx in np.ndindex(B_param.shape):
        orig = B_param[idx]
        B_param[idx] = orig + eps
        y_pos, _ = selective_scan_forward(u, delta, A, B_param, C_param, D=D_skip)
        B_param[idx] = orig - eps
        y_neg, _ = selective_scan_forward(u, delta, A, B_param, C_param, D=D_skip)
        B_param[idx] = orig
        num_dB[idx] = np.sum(d_y * (y_pos - y_neg)) / (2.0 * eps)
    np.testing.assert_allclose(dB, num_dB, atol=1e-5)

    # Check dC
    num_dC = np.zeros_like(C_param)
    for idx in np.ndindex(C_param.shape):
        orig = C_param[idx]
        C_param[idx] = orig + eps
        y_pos, _ = selective_scan_forward(u, delta, A, B_param, C_param, D=D_skip)
        C_param[idx] = orig - eps
        y_neg, _ = selective_scan_forward(u, delta, A, B_param, C_param, D=D_skip)
        C_param[idx] = orig
        num_dC[idx] = np.sum(d_y * (y_pos - y_neg)) / (2.0 * eps)
    np.testing.assert_allclose(dC, num_dC, atol=1e-5)

    # Check dD
    num_dD = np.zeros_like(D_skip)
    for idx in np.ndindex(D_skip.shape):
        orig = D_skip[idx]
        D_skip[idx] = orig + eps
        y_pos, _ = selective_scan_forward(u, delta, A, B_param, C_param, D=D_skip)
        D_skip[idx] = orig - eps
        y_neg, _ = selective_scan_forward(u, delta, A, B_param, C_param, D=D_skip)
        D_skip[idx] = orig
        num_dD[idx] = np.sum(d_y * (y_pos - y_neg)) / (2.0 * eps)
    np.testing.assert_allclose(dD, num_dD, atol=1e-5)




