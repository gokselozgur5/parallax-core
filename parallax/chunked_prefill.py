"""Chunked Prefill Attention for uniform inference scheduling.

Splits long prompt prefill sequences into bounded token chunks, computing causal
attention against both previously cached prefix tokens and causal tokens within
the active chunk.
"""

from typing import Tuple, Optional
import numpy as np


def make_chunked_causal_mask(chunk_len: int, prefix_len: int) -> np.ndarray:
    """Create attention mask for a prefill chunk attending to prefix + intra-chunk.

    Args:
        chunk_len: Number of tokens in the current prefill chunk.
        prefix_len: Number of already cached historical prefix tokens.

    Returns:
        Boolean mask of shape (chunk_len, prefix_len + chunk_len) where True allows attention.
    """
    total_len = prefix_len + chunk_len
    mask = np.ones((chunk_len, total_len), dtype=bool)

    # All prefix tokens [0 : prefix_len] are fully visible to all chunk tokens
    # Intra-chunk tokens must respect causal lower triangular structure
    for i in range(chunk_len):
        future_start = prefix_len + i + 1
        if future_start < total_len:
            mask[i, future_start:] = False

    return mask
