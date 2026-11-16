"""FlashAttention sliding window attention dispatch helpers."""

from __future__ import annotations

import numpy as np


def build_sliding_window_mask(
    seq_len_q: int, seq_len_k: int, window_size: int, causal: bool = True
) -> np.ndarray:
    """Construct boolean mask where valid attention positions are True."""
    if window_size < 0:
        raise ValueError(f"window_size must be non-negative, got {window_size}")
    q_idx = np.arange(seq_len_q)[:, None]
    k_idx = np.arange(seq_len_k)[None, :]
    delta = q_idx - k_idx
    if causal:
        return (delta >= 0) & (delta <= window_size)
    return np.abs(delta) <= window_size


def get_sliding_window_bounds(
    seq_len_q: int, seq_len_k: int, window_size: int, causal: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-query [start, end) key indices for sliding window attention."""
    q_pos = np.arange(seq_len_q)
    starts = np.maximum(0, q_pos - window_size)
    if causal:
        ends = np.minimum(seq_len_k, q_pos + 1)
    else:
        ends = np.minimum(seq_len_k, q_pos + window_size + 1)
    return starts, ends
