"""Precomputed Rotary Positional Embedding (RoPE) dispatch helpers."""

from __future__ import annotations

import numpy as np


def precompute_freqs(
    dim: int, max_seq_len: int, theta: float = 10000.0
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute cos and sin frequency tables up to max_seq_len."""
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"dim must be positive and even, got {dim}")
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
    if theta <= 0.0:
        raise ValueError(f"theta must be positive, got {theta}")
    half = dim // 2
    freqs = 1.0 / (float(theta) ** (np.arange(half, dtype=np.float64) / half))
    pos = np.arange(max_seq_len, dtype=np.float64)[:, None]
    angles = pos * freqs[None, :]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
