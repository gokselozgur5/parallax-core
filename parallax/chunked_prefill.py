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



def chunked_prefill_attention(
    q_chunk: np.ndarray,
    k_all: np.ndarray,
    v_all: np.ndarray,
    prefix_len: int,
) -> np.ndarray:
    """Compute attention for a prefill chunk against historical prefix + intra-chunk keys/values.

    Args:
        q_chunk: Query vectors for chunk, shape (chunk_len, d).
        k_all: All keys up to current chunk end, shape (prefix_len + chunk_len, d).
        v_all: All values up to current chunk end, shape (prefix_len + chunk_len, d).
        prefix_len: Historical prefix length.

    Returns:
        Output representations for chunk tokens, shape (chunk_len, d).
    """
    chunk_len, d = q_chunk.shape
    total_len = k_all.shape[0]
    scale = 1.0 / np.sqrt(d)

    # Raw attention logits: (chunk_len, total_len)
    logits = np.dot(q_chunk, k_all.T) * scale

    # Apply chunked causal mask
    mask = make_chunked_causal_mask(chunk_len=chunk_len, prefix_len=prefix_len)
    logits[~mask] = -1e9

    # Numerically stable softmax
    m = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - m)
    weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return np.dot(weights, v_all)
