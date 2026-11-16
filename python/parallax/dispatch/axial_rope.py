"""2D Axial RoPE dispatch helpers for multimodal and vision models."""

from __future__ import annotations

import numpy as np


def precompute_axial_freqs(
    dim: int, max_h: int, max_w: int, theta: float = 10000.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute 2D axial cos and sin frequencies for vertical and horizontal dimensions."""
    if dim <= 0 or dim % 4 != 0:
        raise ValueError(f"dim must be positive and divisible by 4, got {dim}")
    if max_h <= 0 or max_w <= 0:
        raise ValueError(f"max_h and max_w must be positive, got max_h={max_h}, max_w={max_w}")
    axial_dim = dim // 2
    half = axial_dim // 2
    freqs = 1.0 / (float(theta) ** (np.arange(half, dtype=np.float64) / half))
    h_pos = np.arange(max_h, dtype=np.float64)[:, None]
    w_pos = np.arange(max_w, dtype=np.float64)[:, None]
    cos_h = np.cos(h_pos * freqs[None, :]).astype(np.float32)
    sin_h = np.sin(h_pos * freqs[None, :]).astype(np.float32)
    cos_w = np.cos(w_pos * freqs[None, :]).astype(np.float32)
    sin_w = np.sin(w_pos * freqs[None, :]).astype(np.float32)
    return cos_h, sin_h, cos_w, sin_w
