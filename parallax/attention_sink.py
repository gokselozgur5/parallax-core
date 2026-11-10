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
        if sink_tokens < 0:
            raise ValueError("sink_tokens must be non-negative")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.sink_tokens = sink_tokens
        self.window_size = window_size
        self.max_capacity = sink_tokens + window_size
        self.total_seen = 0
        self._prev_len = 0

    def _update_seen(self, seq_len: int) -> None:
        """Update total seen token count based on incoming sequence length."""
        if self.total_seen == 0:
            delta = seq_len
        else:
            delta = max(0, seq_len - self._prev_len)
        self.total_seen += delta
        self._prev_len = min(seq_len, self.max_capacity)

    def evict(
        self, k_cache: np.ndarray, v_cache: np.ndarray, update_seen: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evict middle tokens if length along seq axis exceeds max_capacity.

        Expected shape: [batch, heads, seq_len, head_dim] (axis=2 is seq_len).
        """
        seq_len = k_cache.shape[2]
        if update_seen:
            self._update_seen(seq_len)

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
        if not kv_caches:
            return kv_caches
        seq_len = kv_caches[0][0].shape[2]
        self._update_seen(seq_len)
        return [self.evict(k, v, update_seen=False) for k, v in kv_caches]

    def reset(self) -> None:
        """Reset total seen count and internal tracking."""
        self.total_seen = 0
        self._prev_len = 0

