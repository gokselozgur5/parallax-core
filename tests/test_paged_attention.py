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


from parallax.paged_attention import SequenceManager


def test_sequence_manager_append_and_block_table_growth():
    alloc = BlockAllocator(num_blocks=4, block_size=2, head_dim=4)
    mgr = SequenceManager(alloc)

    # Append 3 tokens: should allocate 2 blocks (block_size=2)
    for _ in range(3):
        k = np.ones(4)
        v = np.ones(4) * 2
        mgr.append_token(seq_id=1, k_vec=k, v_vec=v)

    assert mgr.seq_lengths[1] == 3
    assert len(mgr.block_tables[1]) == 2
    assert len(alloc.free_blocks) == 2
