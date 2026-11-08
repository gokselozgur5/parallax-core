"""Ring Attention blockwise parallel attention kernel."""

from typing import Optional, Tuple
import numpy as np


class RingAttentionContext:
    """Manages blockwise state for Ring Attention simulation across workers."""

    def __init__(self, block_size: int = 16) -> None:
        self.block_size = int(block_size)
