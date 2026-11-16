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


def apply_rotary_emb(
    x: np.ndarray, cos: np.ndarray, sin: np.ndarray, offset: int = 0, interleaved: bool = False
) -> np.ndarray:
    """Apply precomputed RoPE tables to tensor x."""
    t, half = x.shape[0], x.shape[-1] // 2
    c = cos[offset : offset + t].reshape((t,) + (1,) * (x.ndim - 2) + (half,))
    s = sin[offset : offset + t].reshape((t,) + (1,) * (x.ndim - 2) + (half,))
    out = np.empty_like(x)
    if interleaved:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out[..., 0::2], out[..., 1::2] = x1 * c - x2 * s, x1 * s + x2 * c
    else:
        x1, x2 = x[..., :half], x[..., half:]
        out[..., :half], out[..., half:] = x1 * c - x2 * s, x1 * s + x2 * c
    return out


class RoPEFrequencyTable:
    """Precomputed RoPE frequency table container."""

    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.cos, self.sin = precompute_freqs(dim, max_seq_len, theta=theta)

    def get_slice(self, seq_len: int, offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
        if offset < 0 or offset + seq_len > self.max_seq_len:
            raise IndexError(f"Slice [{offset}:{offset + seq_len}] exceeds max {self.max_seq_len}")
        return self.cos[offset : offset + seq_len], self.sin[offset : offset + seq_len]

    def rotate(self, x: np.ndarray, offset: int = 0, interleaved: bool = False) -> np.ndarray:
        return apply_rotary_emb(x, self.cos, self.sin, offset=offset, interleaved=interleaved)


def dispatch_rope(
    x: np.ndarray,
    start_pos: int = 0,
    theta: float = 10000.0,
    interleaved: bool = True,
) -> np.ndarray:
    """Dispatch RoPE operation to native kernel if available, otherwise numpy."""
    dim = x.shape[-1]
    seq_len = x.shape[0]
    cos, sin = precompute_freqs(dim, max_seq_len=start_pos + seq_len, theta=theta)
    return apply_rotary_emb(x, cos, sin, offset=start_pos, interleaved=interleaved)
