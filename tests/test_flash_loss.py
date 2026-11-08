import numpy as np
from parallax.flash_loss import flash_cross_entropy_forward


def test_flash_cross_entropy_forward_matches_standard():
    np.random.seed(42)
    N, V = 8, 128
    logits = np.random.randn(N, V).astype(np.float32)
    targets = np.random.randint(0, V, size=N)

    # Reference loss calculation
    m = np.max(logits, axis=1, keepdims=True)
    lse = m.squeeze() + np.log(np.sum(np.exp(logits - m), axis=1))
    target_vals = np.array([logits[i, targets[i]] for i in range(N)])
    expected_loss = float(np.mean(lse - target_vals))

    loss, m_out, l_out = flash_cross_entropy_forward(logits, targets, chunk_size=32)

    assert np.isclose(loss, expected_loss, atol=1e-5)
