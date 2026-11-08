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

    def allocate_block(self) -> int:
        """Pop an available block ID from the free list."""
        if not self.free_blocks:
            raise MemoryError("Out of memory: no free physical blocks available")
        return self.free_blocks.pop(0)

    def free_block(self, block_id: int):
        """Return a block ID back to the free list."""
        if block_id in self.free_blocks:
            raise ValueError(f"Block {block_id} is already free")
        self.free_blocks.append(block_id)
