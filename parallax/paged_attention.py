"""PagedAttention memory manager and kernel for zero-fragmentation LLM serving.

Reference: Kwon et al., 2023 - Efficient Memory Management for Large Language Model
Serving with PagedAttention (vLLM).
Partitions KV cache into fixed-size physical blocks and tracks sequences via block tables.
"""

from typing import List, Dict
import numpy as np


class BlockAllocator:
    """Manages physical memory blocks for KV cache."""

    def __init__(self, num_blocks: int, block_size: int, head_dim: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_dim = head_dim
        # Free list tracking available block IDs
        self.free_blocks: List[int] = list(range(num_blocks))
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
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError(f"Block {block_id} out of bounds for num_blocks {self.num_blocks}")
        if block_id in self.free_blocks:
            raise ValueError(f"Block {block_id} is already free")
        self.free_blocks.append(block_id)

    def free_blocks_batch(self, block_ids: List[int]):
        """Return multiple block IDs back to the free list."""
        for block_id in block_ids:
            self.free_block(block_id)



class SequenceManager:
    """Tracks logical tokens to physical block mapping for sequences."""

    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator
        # Mapping from seq_id to list of allocated physical block_ids
        self.block_tables: Dict[int, List[int]] = {}
        # Mapping from seq_id to current sequence length
        self.seq_lengths: Dict[int, int] = {}

    def append_token(self, seq_id: int, k_vec: np.ndarray, v_vec: np.ndarray):
        """Append a new key and value token vector to sequence's paged memory."""
        if seq_id not in self.block_tables:
            self.block_tables[seq_id] = []
            self.seq_lengths[seq_id] = 0

        cur_len = self.seq_lengths[seq_id]
        block_size = self.allocator.block_size
        offset = cur_len % block_size

        if offset == 0:
            # Need a new physical block
            new_block_id = self.allocator.allocate_block()
            self.block_tables[seq_id].append(new_block_id)

        target_block_id = self.block_tables[seq_id][-1]
        self.allocator.k_cache[target_block_id, offset] = k_vec
        self.allocator.v_cache[target_block_id, offset] = v_vec
        self.seq_lengths[seq_id] += 1

    def release_sequence(self, seq_id: int) -> None:
        """Release allocated physical blocks of a sequence back to allocator."""
        if seq_id not in self.block_tables:
            return
        blocks = self.block_tables.pop(seq_id)
        self.seq_lengths.pop(seq_id, None)
        for block_id in blocks:
            self.allocator.free_block(block_id)

    def release(self, seq_id: int) -> None:
        """Alias for release_sequence."""
        self.release_sequence(seq_id)


def paged_attention(
    q: np.ndarray,
    seq_id: int,
    seq_manager: SequenceManager,
) -> np.ndarray:
    """Compute attention output for a query token reading from paged KV cache.

    Args:
        q: Query token vector of shape (d,).
        seq_id: Sequence ID being decoded.
        seq_manager: SequenceManager holding block tables and allocator.

    Returns:
        Output vector of shape (d,).
    """
    total_len = seq_manager.seq_lengths[seq_id]
    if total_len == 0:
        raise ValueError("Cannot attend over empty sequence")

    block_size = seq_manager.allocator.block_size
    head_dim = seq_manager.allocator.head_dim
    scale = 1.0 / np.sqrt(head_dim)
    blocks = seq_manager.block_tables[seq_id]

    # Gather continuous K and V views through the virtual page table
    k_all = np.zeros((total_len, head_dim), dtype=np.float32)
    v_all = np.zeros((total_len, head_dim), dtype=np.float32)

    filled = 0
    for block_id in blocks:
        take = min(block_size, total_len - filled)
        k_all[filled : filled + take] = seq_manager.allocator.k_cache[block_id, :take]
        v_all[filled : filled + take] = seq_manager.allocator.v_cache[block_id, :take]
        filled += take
        if filled >= total_len:
            break

    # Standard scaled dot-product across gathered paged tokens
    logits = np.dot(k_all, q) * scale
    max_logit = np.max(logits)
    weights = np.exp(logits - max_logit)
    weights /= np.sum(weights)

    return np.dot(weights, v_all)
