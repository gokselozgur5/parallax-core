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

def test_paged_flash_decode_exact_match() -> None:
    np.random.seed(42)
    dim = 16
    seq_len = 24
    decoder = PagedFlashDecoder(block_size=8, num_splits=3)
    k_blocks = np.random.randn(4, 8, dim)
    v_blocks = np.random.randn(4, 8, dim)
    block_table = [0, 2, 1]
    q_token = np.random.randn(dim)

    # Reference attention over gathered sequence
    gathered_k, gathered_v = decoder.gather_paged_blocks(
        block_table, k_blocks, v_blocks, seq_len
    )
    scale = 1.0 / np.sqrt(dim)
    ref_scores = np.matmul(q_token, gathered_k.T) * scale
    exp_s = np.exp(ref_scores - np.max(ref_scores))
    weights = exp_s / np.sum(exp_s)
    ref_out = np.matmul(weights, gathered_v)

    decode_out = decoder.decode(q_token, block_table, k_blocks, v_blocks, seq_len)
    np.testing.assert_allclose(decode_out, ref_out, rtol=1e-5, atol=1e-5)
