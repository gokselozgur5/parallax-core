from parallax.contrastive_decoding import ContrastiveDecoder


def test_contrastive_decoder_init() -> None:
    decoder = ContrastiveDecoder(alpha=0.6, beta=0.05)
    assert decoder.alpha == 0.6
    assert decoder.beta == 0.05

import numpy as np


def test_contrastive_logits_masking() -> None:
    decoder = ContrastiveDecoder(alpha=0.5, beta=0.5)
    # Expert strongly favors token 0 (prob ~ 0.88), token 1 has low prob (~0.11)
    expert_logits = np.array([2.0, 0.0, -10.0])
    amateur_logits = np.array([0.0, 0.0, 0.0])
    cd_logits = decoder.compute_contrastive_logits(expert_logits, amateur_logits)
    assert cd_logits.shape == (3,)
    # Token 2 should be masked out to -1e9
    assert cd_logits[2] == -1e9
