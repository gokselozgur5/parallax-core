from parallax.dynamic_batching import SequenceBinner


def test_sequence_binner_init() -> None:
    binner = SequenceBinner(bucket_boundaries=[32, 16, 64, 16])
    assert binner.bucket_boundaries == [16, 32, 64]

def test_find_bucket_snapping() -> None:
    binner = SequenceBinner(bucket_boundaries=[16, 32, 64])
    assert binner.find_bucket(10) == 16
    assert binner.find_bucket(16) == 16
    assert binner.find_bucket(25) == 32
    assert binner.find_bucket(100) == 64

import numpy as np


def test_collate_batch_padding_and_mask() -> None:
    binner = SequenceBinner(bucket_boundaries=[8, 16])
    sequences = [[1, 2, 3], [4, 5, 6, 7, 8]]
    padded, mask = binner.collate_batch(sequences, pad_token_id=0)
    assert padded.shape == (2, 8)
    assert mask.shape == (2, 8)
    assert np.array_equal(mask[0], [1, 1, 1, 0, 0, 0, 0, 0])
    assert np.array_equal(mask[1], [1, 1, 1, 1, 1, 0, 0, 0])
