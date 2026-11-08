"""
YaRN (Yet another RoPE extensioN: Bowen Peng et al., 2023) mathematical primitives.
"""

from typing import Tuple
import numpy as np


def yarn_find_correction_dim(
    num_rotations: float,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 2048,
) -> float:
    """Finds the dimension index corresponding to a specific number of wavelength rotations."""
    return (dim * np.log(max_position_embeddings / (num_rotations * 2 * np.pi))) / (2 * np.log(base))


def yarn_linear_ramp_mask(
    low: float,
    high: float,
    dim: int,
) -> np.ndarray:
    """Computes smooth ramp weights gamma in [0, 1] between low and high interpolation dimensions."""
    if low == high:
        high += 0.001
    linear_func = (np.arange(dim, dtype=np.float64) - low) / (high - low)
    return np.clip(linear_func, 0.0, 1.0)
