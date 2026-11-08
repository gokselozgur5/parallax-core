from parallax.paged_flash_decoding import PagedFlashDecoder


def test_paged_flash_decoder_init() -> None:
    decoder = PagedFlashDecoder(block_size=16, num_splits=4)
    assert decoder.block_size == 16
    assert decoder.num_splits == 4

import numpy as np


def test_gather_paged_blocks_shape() -> None:
    decoder = PagedFlashDecoder(block_size=4)
    # 4 physical blocks of size (4, 8)
    k_blocks = np.zeros((4, 4, 8))
    v_blocks = np.ones((4, 4, 8))
    k_blocks[1] = 5.0
    gathered_k, gathered_v = decoder.gather_paged_blocks(
        block_table=[1, 0], k_blocks=k_blocks, v_blocks=v_blocks, seq_len=6
    )
    assert gathered_k.shape == (6, 8)
    assert np.all(gathered_k[:4] == 5.0)
