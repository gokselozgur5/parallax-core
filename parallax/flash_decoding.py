"""FlashDecoding re-exports delegating to flash_decoding_v2 kernels."""

import numpy as np

from parallax.flash_decoding_v2 import (
    flash_decode_split_k_v2,
    flash_decode_split_kernel,
)


def reduce_split_k_outputs(
    local_outs: np.ndarray,
    local_maxes: np.ndarray,
    local_lsums: np.ndarray,
) -> np.ndarray:
    """Reduce partial split outputs across the K dimension into final attention output."""
    m_global = np.max(local_maxes)
    weights = local_lsums * np.exp(local_maxes - m_global)
    l_global = np.sum(weights)

    scales = np.exp(local_maxes - m_global)[:, np.newaxis]
    rescaled_outs = local_outs * scales

    return np.sum(rescaled_outs, axis=0) / (l_global + 1e-12)


def flash_decode_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    split_size: int = 64,
) -> np.ndarray:
    """FlashDecoding attention forward pass delegating to flash_decode_split_k_v2."""
    seq_len = k.shape[0]
    num_splits = max(1, int(np.ceil(seq_len / split_size))) if split_size > 0 else 1
    return flash_decode_split_k_v2(q, k, v, k_splits=num_splits)


__all__ = [
    "flash_decode_attention",
    "flash_decode_split_kernel",
    "reduce_split_k_outputs",
]
