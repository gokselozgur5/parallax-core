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


def precompute_freqs_cis(
    dim: int, max_seq_len: int, theta: float = 10000.0
) -> np.ndarray:
    """Precompute complex exponential cis (cos + 1j*sin) frequency table."""
    cos, sin = precompute_freqs(dim, max_seq_len, theta=theta)
    return (cos + 1j * sin).astype(np.complex64)


def rotate_half(x: np.ndarray, interleaved: bool = False) -> np.ndarray:
    """Rotate half the hidden dimensions of x."""
    if interleaved:
        out = np.empty_like(x)
        out[..., 0::2] = -x[..., 1::2]
        out[..., 1::2] = x[..., 0::2]
        return out
    half = x.shape[-1] // 2
    return np.concatenate((-x[..., half:], x[..., :half]), axis=-1)
