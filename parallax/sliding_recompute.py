"""Sliding Window Recomputation attention kernel."""

import numpy as np


class SlidingWindowBuffer:
    """Manages active attention sinks and sliding window tokens."""

    def __init__(self, window_size: int, sink_size: int = 2) -> None:
        self.window_size = int(window_size)
        self.sink_size = int(sink_size)

    def filter_active_indices(self, total_seq_len: int) -> np.ndarray:
        """Return indices of tokens retained by sinks + sliding window."""
        if total_seq_len <= self.sink_size + self.window_size:
            return np.arange(total_seq_len)
        sinks = np.arange(self.sink_size)
        window = np.arange(total_seq_len - self.window_size, total_seq_len)
        return np.concatenate([sinks, window])

    def recompute_attention(
        self,
        q_token: np.ndarray,
        all_k: np.ndarray,
        all_v: np.ndarray,
    ) -> np.ndarray:
        """Recompute attention of new query against only active sink and window KV pairs."""
        seq_len, dim = all_k.shape
        active_idx = self.filter_active_indices(seq_len)
        active_k = all_k[active_idx]
        active_v = all_v[active_idx]
        scale = 1.0 / np.sqrt(dim)
        scores = np.matmul(q_token, active_k.T) * scale
        exp_s = np.exp(scores - np.max(scores))
        weights = exp_s / np.sum(exp_s)
        return np.matmul(weights, active_v)
