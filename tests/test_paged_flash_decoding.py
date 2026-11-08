from parallax.paged_flash_decoding import PagedFlashDecoder


def test_paged_flash_decoder_init() -> None:
    decoder = PagedFlashDecoder(block_size=16, num_splits=4)
    assert decoder.block_size == 16
    assert decoder.num_splits == 4
