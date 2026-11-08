from parallax.contrastive_decoding import ContrastiveDecoder


def test_contrastive_decoder_init() -> None:
    decoder = ContrastiveDecoder(alpha=0.6, beta=0.05)
    assert decoder.alpha == 0.6
    assert decoder.beta == 0.05
