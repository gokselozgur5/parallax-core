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


def precompute_axial_freqs_cis(
    dim: int, max_h: int, max_w: int, theta: float = 10000.0
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute complex exponential cis tables for vertical and horizontal dimensions."""
    cos_h, sin_h, cos_w, sin_w = precompute_axial_freqs(dim, max_h, max_w, theta=theta)
    cis_h = (cos_h + 1j * sin_h).astype(np.complex64)
    cis_w = (cos_w + 1j * sin_w).astype(np.complex64)
    return cis_h, cis_w


def _rot(sub_x: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    half = sub_x.shape[-1] // 2
    x1, x2 = sub_x[..., :half], sub_x[..., half:]
    c = c.reshape((sub_x.shape[0],) + (1,) * (sub_x.ndim - 2) + (half,))
    s = s.reshape((sub_x.shape[0],) + (1,) * (sub_x.ndim - 2) + (half,))
    return np.concatenate((x1 * c - x2 * s, x1 * s + x2 * c), axis=-1)


def apply_axial_rope(
    x: np.ndarray, cos_h: np.ndarray, sin_h: np.ndarray, cos_w: np.ndarray, sin_w: np.ndarray, h_coords: np.ndarray, w_coords: np.ndarray
) -> np.ndarray:
    """Apply 2D axial rotary embedding to x along vertical and horizontal coordinates."""
    axial_dim = x.shape[-1] // 2
    x_h = _rot(x[..., :axial_dim], cos_h[h_coords], sin_h[h_coords])
    x_w = _rot(x[..., axial_dim:], cos_w[w_coords], sin_w[w_coords])
    return np.concatenate((x_h, x_w), axis=-1)


def apply_axial_rope_grid(
    x: np.ndarray, cos_h: np.ndarray, sin_h: np.ndarray, cos_w: np.ndarray, sin_w: np.ndarray, height: int, width: int
) -> np.ndarray:
    """Apply 2D axial rotary embedding to a flattened grid sequence."""
    if x.shape[0] != height * width:
        raise ValueError(f"Sequence length {x.shape[0]} does not match grid {height}x{width}")
    h_coords = np.repeat(np.arange(height), width)
    w_coords = np.tile(np.arange(width), height)
    return apply_axial_rope(x, cos_h, sin_h, cos_w, sin_w, h_coords, w_coords)


class AxialRoPEFrequencyTable:
    """Precomputed 2D axial RoPE frequency table container."""

    def __init__(self, dim: int, max_h: int, max_w: int, theta: float = 10000.0) -> None:
        self.dim = dim
        self.max_h = max_h
        self.max_w = max_w
        self.theta = theta
        self.cos_h, self.sin_h, self.cos_w, self.sin_w = precompute_axial_freqs(
            dim, max_h, max_w, theta=theta
        )

    def apply_to_grid(self, x: np.ndarray, height: int, width: int) -> np.ndarray:
        return apply_axial_rope_grid(x, self.cos_h, self.sin_h, self.cos_w, self.sin_w, height, width)


def dispatch_axial_rope(
    x: np.ndarray, height: int, width: int, theta: float = 10000.0
) -> np.ndarray:
    """Dispatch 2D axial rotary embedding onto tensor x with grid (height, width)."""
    dim = x.shape[-1]
    cos_h, sin_h, cos_w, sin_w = precompute_axial_freqs(dim, height, width, theta=theta)
    return apply_axial_rope_grid(x, cos_h, sin_h, cos_w, sin_w, height, width)


__all__ = [
    "AxialRoPEFrequencyTable",
    "apply_axial_rope",
    "apply_axial_rope_grid",
    "dispatch_axial_rope",
    "precompute_axial_freqs",
    "precompute_axial_freqs_cis",
]
