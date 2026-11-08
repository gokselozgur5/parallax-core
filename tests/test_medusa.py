import numpy as np
from parallax.medusa import MedusaHead


def test_medusa_head_initialization_and_forward():
    np.random.seed(42)
    head = MedusaHead(hidden_dim=32, vocab_size=100, offset=1)
    hidden = np.random.randn(32).astype(np.float32)
    logits = head.forward(hidden)
    assert logits.shape == (100,)
    assert not np.isnan(logits).any()


from parallax.medusa import MedusaModel


def test_medusa_model_predict_candidates():
    np.random.seed(42)
    model = MedusaModel(hidden_dim=16, vocab_size=50, num_heads=4)
    hidden = np.random.randn(16).astype(np.float32)
    cands = model.predict_candidates(hidden)
    assert len(cands) == 4
    for tok in cands:
        assert 0 <= tok < 50


from parallax.medusa import verify_medusa_candidates


def test_verify_medusa_candidates_prefix_match():
    # Ground truth sequence generator: token n is (len % 10)
    def oracle(seq):
        return len(seq) % 10

    prefix = [0, 1]
    # Length is 2 -> oracle returns 2
    # Next len is 3 -> oracle returns 3
    # Next len is 4 -> oracle returns 4
    candidates = [2, 3, 9]  # 9 diverges from oracle's 4

    accepted, next_tok = verify_medusa_candidates(candidates, oracle, prefix)
    assert accepted == [2, 3]
    assert next_tok == 4
