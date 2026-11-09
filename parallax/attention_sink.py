"""
StreamingLLM Attention Sink and Rolling KV Cache Eviction.

Provides bounded-memory autoregressive generation by retaining initial
attention sink tokens and evicting older middle tokens while preserving
the rolling recent local window.
"""

from typing import List, Optional, Tuple
import numpy as np


class RollingKVCache:
    """Rolling KV cache with attention sinks (StreamingLLM).

    Retains the first `sink_tokens` (which absorb high attention mass) and the
    most recent `window_size` tokens, evicting middle tokens when total length
    exceeds `sink_tokens + window_size`.
    """

    def __init__(self, sink_tokens: int = 4, window_size: int = 128):
        assert sink_tokens >= 0, "sink_tokens must be non-negative"
        assert window_size > 0, "window_size must be positive"
        self.sink_tokens = sink_tokens
        self.window_size = window_size
        self.max_capacity = sink_tokens + window_size
        self.total_seen = 0

    def evict(
        self, k_cache: np.ndarray, v_cache: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evict middle tokens if length along seq axis exceeds max_capacity.

        Expected shape: [batch, heads, seq_len, head_dim] (axis=2 is seq_len).
        """
        seq_len = k_cache.shape[2]
        if seq_len <= self.max_capacity:
            return k_cache, v_cache

        # Keep first sink_tokens and last window_size tokens
        if self.sink_tokens > 0:
            sink_k = k_cache[:, :, :self.sink_tokens, :]
            sink_v = v_cache[:, :, :self.sink_tokens, :]
            recent_k = k_cache[:, :, -self.window_size:, :]
            recent_v = v_cache[:, :, -self.window_size:, :]
            k_out = np.concatenate([sink_k, recent_k], axis=2)
            v_out = np.concatenate([sink_v, recent_v], axis=2)
        else:
            k_out = k_cache[:, :, -self.window_size:, :]
            v_out = v_cache[:, :, -self.window_size:, :]

        return k_out, v_out

    def evict_layers(
        self, kv_caches: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Apply rolling eviction across all transformer layer KV caches."""
        return [self.evict(k, v) for k, v in kv_caches]

