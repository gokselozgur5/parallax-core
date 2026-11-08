"""Sliding Window Recomputation attention kernel."""

from typing import Tuple
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
