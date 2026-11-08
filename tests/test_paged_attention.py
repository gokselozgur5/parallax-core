import numpy as np
from parallax.paged_attention import BlockAllocator


def test_block_allocator_initialization():
    alloc = BlockAllocator(num_blocks=10, block_size=16, head_dim=64)
    assert len(alloc.free_blocks) == 10
    assert alloc.k_cache.shape == (10, 16, 64)
    assert alloc.v_cache.shape == (10, 16, 64)


def test_block_allocator_allocation_and_free():
    alloc = BlockAllocator(num_blocks=2, block_size=4, head_dim=8)
    b0 = alloc.allocate_block()
    b1 = alloc.allocate_block()
    assert b0 == 0 and b1 == 1
    assert len(alloc.free_blocks) == 0

    alloc.free_block(b0)
    assert len(alloc.free_blocks) == 1
    b2 = alloc.allocate_block()
    assert b2 == 0
