"""PagedAttention memory manager and kernel for zero-fragmentation LLM serving.

Reference: Kwon et al., 2023 - Efficient Memory Management for Large Language Model
Serving with PagedAttention (vLLM).
Partitions KV cache into fixed-size physical blocks and tracks sequences via block tables.
"""

from typing import List, Tuple, Dict
import numpy as np


class BlockAllocator:
    """Manages physical memory blocks for KV cache."""

    def __init__(self, num_blocks: int, block_size: int, head_dim: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_dim = head_dim
        # Free list tracking available block IDs
        self.free_blocks = list(range(num_blocks))
        # Physical key and value storage: (num_blocks, block_size, head_dim)
        self.k_cache = np.zeros((num_blocks, block_size, head_dim), dtype=np.float32)
        self.v_cache = np.zeros((num_blocks, block_size, head_dim), dtype=np.float32)
