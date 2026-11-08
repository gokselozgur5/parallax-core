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
