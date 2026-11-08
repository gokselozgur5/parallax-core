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


def reduce_split_k_outputs(
    local_outs: np.ndarray,
    local_maxes: np.ndarray,
    local_lsums: np.ndarray,
) -> np.ndarray:
    """Reduce partial split outputs across the K dimension into final attention output.

    Args:
        local_outs: Local partial outputs of shape (num_splits, d).
        local_maxes: Local max logits of shape (num_splits,).
        local_lsums: Local sum-exps of shape (num_splits,).

    Returns:
        Final combined output vector of shape (d,).
    """
    m_global = np.max(local_maxes)
    # Rescale local sums: l_i * exp(m_i - m_global)
    weights = local_lsums * np.exp(local_maxes - m_global)
    l_global = np.sum(weights)

    # Rescale local outputs: O_i * exp(m_i - m_global)
    scales = np.exp(local_maxes - m_global)[:, np.newaxis]
    rescaled_outs = local_outs * scales

    return np.sum(rescaled_outs, axis=0) / (l_global + 1e-12)


def flash_decode_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    split_size: int = 64,
) -> np.ndarray:
    """Full FlashDecoding attention forward pass for single query token across long context.

    Args:
        q: Query token vector of shape (d,) or (1, d).
        k: Cached keys matrix of shape (seq_len, d).
        v: Cached values matrix of shape (seq_len, d).
        split_size: Number of tokens per split chunk.

    Returns:
        Output vector of shape (d,).
    """
    q_vec = q.squeeze()
    seq_len, d = k.shape
    scale = 1.0 / np.sqrt(d)

    num_splits = int(np.ceil(seq_len / split_size))
    local_outs = np.zeros((num_splits, d), dtype=np.float64)
    local_maxes = np.zeros(num_splits, dtype=np.float64)
    local_lsums = np.zeros(num_splits, dtype=np.float64)

    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = min(start_idx + split_size, seq_len)
        k_chunk = k[start_idx:end_idx]
        v_chunk = v[start_idx:end_idx]
        out_i, m_i, l_i = flash_decode_split_kernel(q_vec, k_chunk, v_chunk, scale)
        local_outs[i] = out_i
        local_maxes[i] = m_i
        local_lsums[i] = l_i

    return reduce_split_k_outputs(local_outs, local_maxes, local_lsums)
