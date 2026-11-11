"""FlashDecoding-v2 module with Dynamic Split-K and Paged KV-Cache integration.

Optimized for single-token generation across long sequence contexts.
Features:
- Dynamic Split-K partition selection scaling with sequence length (K_splits=1, 2, 4, 8, etc.)
- Numerically stable online reduction across splits using logsumexp / max-rescaling
- Seamless Paged KV-Cache integration via flash_decode_split_k_paged
"""

from typing import List, Optional, Tuple, Union
import numpy as np


def select_split_k(seq_len: int, min_chunk_size: int = 64, max_splits: int = 8) -> int:
    """Dynamically determine the Split-K partition factor based on sequence length.

    Scales K_splits to 1, 2, 4, 8 depending on sequence length to balance
    parallel reduction overhead against compute throughput.

    Args:
        seq_len: Total sequence length of cached keys/values.
        min_chunk_size: Minimum number of tokens per split chunk.
        max_splits: Upper bound for split factor (e.g. 8).

    Returns:
        Chosen integer partition count (power of two or clamped integer >= 1).
    """
    if seq_len <= 0 or min_chunk_size <= 0:
        return 1

    max_possible = max(1, seq_len // min_chunk_size)
    splits = 1
    while splits * 2 <= max_possible and splits * 2 <= max_splits:
        splits *= 2

    return min(splits, max_splits)


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
        Tuple of (local_out, m_local, l_local):
            local_out: Weighted value sum of shape (d,) using unnormalized exp.
            m_local: Maximum logit within this split (scalar float).
            l_local: Sum of exp(logits - m_local) for this split (scalar float).
    """
    if k_chunk.shape[0] == 0:
        d = q.shape[-1]
        return np.zeros(d, dtype=np.float64), -float("inf"), 0.0

    logits = np.dot(k_chunk, q) * scale
    m_local = float(np.max(logits))
    p = np.exp(logits - m_local)
    l_local = float(np.sum(p))
    local_out = np.dot(p, v_chunk)
    return local_out, m_local, l_local


def reduce_split_k_online(
    local_outs: np.ndarray,
    local_maxes: np.ndarray,
    local_lsums: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Numerically stable online reduction across Split-K partitions.

    Uses logsumexp / maximum alignment to reduce partial partition outputs
    into the combined attention output vector without numerical overflow.

    Reduction formula:
        m_global = max_i(m_i)
        l_global = sum_i(l_i * exp(m_i - m_global))
        O_global = sum_i(O_i * exp(m_i - m_global)) / l_global
        logsumexp = m_global + log(l_global)

    Args:
        local_outs: Array of local outputs of shape (num_splits, d).
        local_maxes: Array of local max logits of shape (num_splits,).
        local_lsums: Array of local sum-exps of shape (num_splits,).

    Returns:
        Tuple of (final_output, global_logsumexp).
    """
    valid_mask = np.isfinite(local_maxes) & (local_lsums > 0.0)
    if not np.any(valid_mask):
        d = local_outs.shape[-1]
        return np.zeros(d, dtype=np.float64), -float("inf")

    filtered_outs = local_outs[valid_mask]
    filtered_maxes = local_maxes[valid_mask]
    filtered_lsums = local_lsums[valid_mask]

    m_global = float(np.max(filtered_maxes))
    factors = np.exp(filtered_maxes - m_global)

    weights = filtered_lsums * factors
    l_global = float(np.sum(weights))

    scales = factors[:, np.newaxis]
    rescaled_outs = filtered_outs * scales
    summed_out = np.sum(rescaled_outs, axis=0)

    final_out = summed_out / max(l_global, 1e-12)
    global_lse = m_global + float(np.log(max(l_global, 1e-12)))

    return final_out, global_lse


def flash_decode_split_k_v2(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    k_splits: Optional[int] = None,
    min_chunk_size: int = 64,
    scale: Optional[float] = None,
) -> np.ndarray:
    """FlashDecoding-v2 attention for a single query token across a key/value sequence.

    Args:
        q: Query token vector of shape (d,) or (1, d).
        k: Key sequence matrix of shape (seq_len, d).
        v: Value sequence matrix of shape (seq_len, d).
        k_splits: Number of splits. If None, dynamically selected based on seq_len.
        min_chunk_size: Minimum tokens per chunk for dynamic selection.
        scale: Softmax scaling factor. Defaults to 1.0 / sqrt(d).

    Returns:
        Output vector of shape (d,).
    """
    q_vec = q.squeeze()
    seq_len, d = k.shape
    if scale is None:
        scale = 1.0 / np.sqrt(d)

    if seq_len == 0:
        return np.zeros(d, dtype=np.float64)

    if k_splits is None:
        k_splits = select_split_k(seq_len, min_chunk_size=min_chunk_size)

    k_splits = max(1, min(k_splits, seq_len))
    chunk_size = (seq_len + k_splits - 1) // k_splits

    local_outs = []
    local_maxes = []
    local_lsums = []

    for i in range(k_splits):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, seq_len)
        if start_idx >= seq_len:
            continue
        k_chunk = k[start_idx:end_idx]
        v_chunk = v[start_idx:end_idx]
        out_i, m_i, l_i = flash_decode_split_kernel(q_vec, k_chunk, v_chunk, scale)
        local_outs.append(out_i)
        local_maxes.append(m_i)
        local_lsums.append(l_i)

    out, _ = reduce_split_k_online(
        np.array(local_outs, dtype=np.float64),
        np.array(local_maxes, dtype=np.float64),
        np.array(local_lsums, dtype=np.float64),
    )
    return out


def flash_decode_split_k_paged(
    q: np.ndarray,
    block_table: Union[List[int], np.ndarray],
    k_blocks: np.ndarray,
    v_blocks: np.ndarray,
    seq_len: int,
    k_splits: Optional[int] = None,
    min_chunk_size: int = 64,
    scale: Optional[float] = None,
) -> np.ndarray:
    """FlashDecoding-v2 kernel directly operating over Paged KV-Cache blocks.

    Collects the active sequence slices from the physical block pool according
    to the logical-to-physical block table, applies dynamic Split-K partitioning,
    and performs numerically stable online logsumexp reduction.

    Args:
        q: Query token vector of shape (d,) or (1, d).
        block_table: Ordered list or 1D array of physical block indices.
        k_blocks: Physical Key cache pool of shape (num_blocks, block_size, d).
        v_blocks: Physical Value cache pool of shape (num_blocks, block_size, d).
        seq_len: Number of valid tokens in this sequence.
        k_splits: Optional partition factor. If None, dynamically selected.
        min_chunk_size: Minimum tokens per chunk for dynamic Split-K.
        scale: Softmax scale factor. Defaults to 1.0 / sqrt(d).

    Returns:
        Output vector of shape (d,).
    """
    q_vec = q.squeeze()
    d = q_vec.shape[-1]
    if scale is None:
        scale = 1.0 / np.sqrt(d)

    if seq_len <= 0:
        return np.zeros(d, dtype=np.float64)

    # Gather contiguous sequence up to seq_len
    gathered_k = np.concatenate([k_blocks[b] for b in block_table], axis=0)[:seq_len]
    gathered_v = np.concatenate([v_blocks[b] for b in block_table], axis=0)[:seq_len]

    return flash_decode_split_k_v2(
        q=q_vec,
        k=gathered_k,
        v=gathered_v,
        k_splits=k_splits,
        min_chunk_size=min_chunk_size,
        scale=scale,
    )
