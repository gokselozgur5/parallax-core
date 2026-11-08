"""
Selective State Space Model (S6 / Mamba: Gu & Dao, 2023) mathematical primitives.
"""

from typing import Tuple
import numpy as np


def discretize_zoh(
    delta: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-Order Hold (ZOH) discretization for selective continuous-time state equations.

    A_bar = exp(delta * A)
    B_bar = delta * B
    """
    # delta: [B, T, D], A: [D, N] -> delta[:, :, :, None] * A[None, None, :, :]
    A_bar = np.exp(delta[:, :, :, None] * A[None, None, :, :])
    # B: [B, T, N] -> B_bar: [B, T, D, N]
    B_bar = delta[:, :, :, None] * B[:, :, None, :]
    return A_bar, B_bar


def selective_scan_forward(
    u: np.ndarray,
    delta: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Selective scan forward pass: h_t = A_bar_t * h_{t-1} + B_bar_t * u_t, y_t = C_t * h_t + D * u_t."""
    batch_size, seq_len, d_inner = u.shape
    d_state = A.shape[1]

    A_bar, B_bar = discretize_zoh(delta, A, B)

    y = np.zeros((batch_size, seq_len, d_inner), dtype=np.float64)
    h_states = np.zeros((batch_size, seq_len, d_inner, d_state), dtype=np.float64)
    h_curr = np.zeros((batch_size, d_inner, d_state), dtype=np.float64)

    for t in range(seq_len):
        # h_curr = A_bar[:, t] * h_curr + B_bar[:, t] * u[:, t, :, None]
        h_curr = A_bar[:, t] * h_curr + B_bar[:, t] * u[:, t, :, None]
        h_states[:, t] = h_curr

        # y_t = sum_n (C_t[:, None, :] * h_curr) -> [B, D]
        y[:, t] = np.sum(C[:, t, None, :] * h_curr, axis=-1)

    if D is not None:
        y += u * D[None, None, :]

    return y, h_states

