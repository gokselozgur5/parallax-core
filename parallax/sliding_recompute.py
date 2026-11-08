"""Sliding Window Recomputation attention kernel."""

from typing import Tuple
import numpy as np


class SlidingWindowBuffer:
    """Manages active attention sinks and sliding window tokens."""

    def __init__(self, window_size: int, sink_size: int = 2) -> None:
        self.window_size = int(window_size)
        self.sink_size = int(sink_size)
