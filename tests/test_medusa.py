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
