"""
YaRN (Yet another RoPE extensioN: Bowen Peng et al., 2023) mathematical primitives.
"""

from typing import Tuple
import numpy as np


def yarn_find_correction_dim(
    num_rotations: float,
    half_dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 2048,
) -> float:
    """Finds the dimension index corresponding to a specific number of wavelength rotations."""
    return (half_dim * np.log(max_position_embeddings / (num_rotations * 2 * np.pi))) / np.log(base)


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


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """Computes attention temperature scaling factor sqrt(1 / t) to counter entropy decay."""
    if scale <= 1.0:
        return 1.0
    return 0.1 * np.log(scale) + 1.0


def apply_yarn_rope(
    x: np.ndarray,
    scale_factor: float = 4.0,
    seq_dim: int = 1,
    offset: int = 0,
    base: float = 10000.0,
    max_position_embeddings: int = 256,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> np.ndarray:
    """Applies YaRN frequency-interpolated RoPE to an ndarray."""
    dim = x.shape[-1]
    if dim % 2 != 0:
        raise ValueError("YaRN requires even head dimension")
    half_dim = dim // 2

    # Standard inverse frequencies
    pos_idx = np.arange(half_dim, dtype=np.float64)
    inv_freq = 1.0 / (base ** (pos_idx / half_dim))

    if scale_factor > 1.0:
        low = yarn_find_correction_dim(beta_fast, half_dim, base, max_position_embeddings)
        high = yarn_find_correction_dim(beta_slow, half_dim, base, max_position_embeddings)
        ramp = yarn_linear_ramp_mask(low, high, half_dim)

        # Interpolate: (1 - gamma) * inv_freq + gamma * (inv_freq / s)
        inv_freq_interpolated = inv_freq / scale_factor
        inv_freq = (1.0 - ramp) * inv_freq + ramp * inv_freq_interpolated

    seq_len = x.shape[seq_dim]
    positions = (np.arange(seq_len, dtype=np.float64) + float(offset))[:, None]
    angles = positions * inv_freq[None, :]

    sin = np.sin(angles)
    cos = np.cos(angles)

    if x.ndim == 4:
        sin, cos = sin[None, :, None, :], cos[None, :, None, :]
    elif x.ndim == 3:
        sin, cos = sin[None, :, :], cos[None, :, :]

    x1, x2 = x[..., 0::2], x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos

    out = np.empty_like(x)
    out[..., 0::2] = rx1
    out[..., 1::2] = rx2

    mscale = yarn_get_mscale(scale_factor)
    return out * mscale


