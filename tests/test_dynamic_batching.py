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
