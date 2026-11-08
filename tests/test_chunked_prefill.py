from parallax.chunked_prefill import make_chunked_causal_mask
import numpy as np


def test_make_chunked_causal_mask_structure():
    chunk_len = 3
    prefix_len = 2
    mask = make_chunked_causal_mask(chunk_len=chunk_len, prefix_len=prefix_len)

    assert mask.shape == (3, 5)
    # First 2 columns (prefix) must be fully True for all chunk rows
    assert np.all(mask[:, :2])
    # Intra-chunk causality: row 0 attends to col 2, but not 3, 4
    assert mask[0, 2] and not mask[0, 3] and not mask[0, 4]
    # Row 1 attends to col 2, 3, but not 4
    assert mask[1, 2] and mask[1, 3] and not mask[1, 4]
    # Row 2 attends to all cols 2, 3, 4
    assert mask[2, 2] and mask[2, 3] and mask[2, 4]
