"""FlashAttention sliding window attention dispatch helpers."""

from __future__ import annotations

import numpy as np


def build_sliding_window_mask(
    seq_len_q: int, seq_len_k: int, window_size: int, causal: bool = True
) -> np.ndarray:
    """Construct boolean mask where valid attention positions are True."""
    if window_size < 0:
        raise ValueError(f"window_size must be non-negative, got {window_size}")
    offset = (seq_len_k - seq_len_q) if causal else 0
    q_idx = (np.arange(seq_len_q) + offset)[:, None]
    k_idx = np.arange(seq_len_k)[None, :]
    delta = q_idx - k_idx
    if causal:
        return (delta >= 0) & (delta <= window_size)
    return np.abs(delta) <= window_size


def get_sliding_window_bounds(
    seq_len_q: int, seq_len_k: int, window_size: int, causal: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-query [start, end) key indices for sliding window attention."""
    offset = (seq_len_k - seq_len_q) if causal else 0
    q_pos = np.arange(seq_len_q) + offset
    starts = np.maximum(0, q_pos - window_size)
    if causal:
        ends = np.minimum(seq_len_k, q_pos + 1)
    else:
        ends = np.minimum(seq_len_k, q_pos + window_size + 1)
    return starts, ends


def sliding_window_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    window_size: int,
    scale: float | None = None,
    causal: bool = True,
) -> np.ndarray:
    """Compute scaled dot-product attention with sliding window masking."""
    ndim3 = q.ndim == 3
    q_b = q[None, ...] if ndim3 else q
    k_b = k[None, ...] if ndim3 else k
    v_b = v[None, ...] if ndim3 else v
    inv_scale = float(scale) if scale is not None else 1.0 / np.sqrt(q_b.shape[-1])
    mask = build_sliding_window_mask(q_b.shape[1], k_b.shape[1], window_size, causal=causal)
    scores = np.einsum("bshd,bthd->bhst", q_b, k_b) * inv_scale
    scores = np.where(mask[None, None, :, :], scores, -1e9)
    probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probs = probs / np.maximum(np.sum(probs, axis=-1, keepdims=True), 1e-12)
    out = np.einsum("bhst,bthd->bshd", probs, v_b).astype(q.dtype)
    return out[0] if ndim3 else out


def flash_sliding_window_tile(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    q_start: int,
    k_start: int,
    window_size: int,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process a single FlashAttention tile with sliding window masking."""
    scores = np.matmul(q, k.T) * scale
    q_idx = np.arange(q_start, q_start + q.shape[0])[:, None]
    k_idx = np.arange(k_start, k_start + k.shape[0])[None, :]
    delta = q_idx - k_idx
    mask = (delta >= 0) & (delta <= window_size)
    scores = np.where(mask, scores, -1e9)
    m = np.max(scores, axis=-1, keepdims=True)
    p = np.exp(scores - m) * mask
    l = np.sum(p, axis=-1, keepdims=True)
    acc = np.matmul(p, v)
    return acc, m, l


class SlidingWindowAttention:
    """Sliding Window Attention layer with optional causal masking."""

    def __init__(
        self, window_size: int, scale: float | None = None, causal: bool = True
    ) -> None:
        if window_size < 0:
            raise ValueError(f"window_size must be >= 0, got {window_size}")
        self.window_size = int(window_size)
        self.scale = scale
        self.causal = causal

    def __call__(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        return sliding_window_attention(
            q, k, v, window_size=self.window_size, scale=self.scale, causal=self.causal
        )


def dispatched_sliding_window_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    window_size: int,
    scale: float | None = None,
    causal: bool = True,
) -> np.ndarray:
    """Dispatched entry point routing sliding window attention computation."""
    return sliding_window_attention(
        q, k, v, window_size=window_size, scale=scale, causal=causal
    )
