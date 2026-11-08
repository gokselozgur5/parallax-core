import numpy as np
from parallax.medusa import MedusaHead


def test_medusa_head_initialization_and_forward():
    np.random.seed(42)
    head = MedusaHead(hidden_dim=32, vocab_size=100, offset=1)
    hidden = np.random.randn(32).astype(np.float32)
    logits = head.forward(hidden)
    assert logits.shape == (100,)
    assert not np.isnan(logits).any()
