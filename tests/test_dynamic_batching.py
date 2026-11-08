from parallax.dynamic_batching import SequenceBinner


def test_sequence_binner_init() -> None:
    binner = SequenceBinner(bucket_boundaries=[32, 16, 64, 16])
    assert binner.bucket_boundaries == [16, 32, 64]
