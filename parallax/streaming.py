"""
Sliding Window Attention with Attention Sinks (StreamingLLM: Xiao et al., 2023).
"""

import numpy as np


def generate_sink_window_mask(
    seq_len: int,
    window_size: int = 16,
    num_sinks: int = 4,
) -> np.ndarray:
    """Constructs a boolean attention mask for StreamingLLM with sink tokens."""
    q_idx = np.arange(seq_len)[:, None]
    k_idx = np.arange(seq_len)[None, :]

    causal_mask = q_idx >= k_idx
    sink_mask = k_idx < num_sinks
    window_mask = (q_idx - k_idx) < window_size

    # Valid if causal AND (is a sink token OR within sliding window)
    valid = causal_mask & (sink_mask | window_mask)
    return valid


def streaming_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    window_size: int = 16,
    num_sinks: int = 4,
) -> np.ndarray:
    """Computes Multi-Head Attention masked with StreamingLLM sink and window constraints."""
    B, H, T, d = q.shape
    scale = 1.0 / np.sqrt(d)
    scores = np.matmul(q, k.swapaxes(-2, -1)) * scale

    valid_mask = generate_sink_window_mask(T, window_size=window_size, num_sinks=num_sinks)
    scores = np.where(valid_mask[None, None, :, :], scores, -1e9)

    max_s = np.max(scores, axis=-1, keepdims=True)
    exp_s = np.exp(scores - max_s)
    probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    return np.matmul(probs, v)


def evict_kv_cache_with_sinks(
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    window_size: int = 16,
    num_sinks: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evicts intermediate keys and values while preserving attention sinks and recent window."""
    # Shape: [B, H, T, d]
    T = k_cache.shape[2]
    max_allowed = num_sinks + window_size
    if T <= max_allowed:
        return k_cache, v_cache

    sink_k, sink_v = k_cache[:, :, :num_sinks, :], v_cache[:, :, :num_sinks, :]
    recent_k, recent_v = k_cache[:, :, -window_size:, :], v_cache[:, :, -window_size:, :]

    new_k = np.concatenate([sink_k, recent_k], axis=2)
    new_v = np.concatenate([sink_v, recent_v], axis=2)
    return new_k, new_v


