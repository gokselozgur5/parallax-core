"""FlashDecoding (Split-K Attention) module for fast long-context decoding.

Reference: Dao et al., 2023 - Flash-Decoding for long-context LLM inference.
Splits the key/value sequence dimension into multiple parallel chunks (splits),
computes partial softmax stats per split, and reduces across splits.
"""

from typing import Tuple
import numpy as np


def flash_decode_split_kernel(
    q: np.ndarray,
    k_chunk: np.ndarray,
    v_chunk: np.ndarray,
    scale: float,
) -> Tuple[np.ndarray, float, float]:
    """Compute local attention numerator, local max, and local sum-exp for one split.

    Args:
        q: Query vector of shape (d,).
        k_chunk: Key chunk of shape (chunk_len, d).
        v_chunk: Value chunk of shape (chunk_len, d).
        scale: Softmax scale factor (1.0 / sqrt(d)).

    Returns:
        Tuple of:
            - local_out: Weighted sum of shape (d,) using unnormalized exp.
            - m_local: Max logit across this split (scalar float).
            - l_local: Sum of exp(logits - m_local) for this split (scalar float).
    """
    logits = np.dot(k_chunk, q) * scale
    m_local = float(np.max(logits))
    p = np.exp(logits - m_local)
    l_local = float(np.sum(p))
    local_out = np.dot(p, v_chunk)
    return local_out, m_local, l_local
