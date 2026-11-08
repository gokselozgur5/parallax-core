import numpy as np
from parallax.paged_attention import BlockAllocator


def test_block_allocator_initialization():
    alloc = BlockAllocator(num_blocks=10, block_size=16, head_dim=64)
    assert len(alloc.free_blocks) == 10
    assert alloc.k_cache.shape == (10, 16, 64)
    assert alloc.v_cache.shape == (10, 16, 64)
