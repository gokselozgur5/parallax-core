"""Generalized FlashDecoding kernel with Paged KV table indexing."""

from typing import List, Tuple
import numpy as np


class PagedFlashDecoder:
    """Combines Split-K FlashDecoding with virtual paged KV block tables."""

    def __init__(self, block_size: int = 16, num_splits: int = 4) -> None:
        self.block_size = int(block_size)
        self.num_splits = int(num_splits)
