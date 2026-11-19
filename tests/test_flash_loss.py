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


from parallax.flash_loss import flash_cross_entropy_backward


def test_flash_cross_entropy_backward_matches_standard():
    np.random.seed(42)
    N, V = 4, 64
    logits = np.random.randn(N, V).astype(np.float32)
    targets = np.random.randint(0, V, size=N)

    # Reference backward
    m = np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits - m) / np.sum(np.exp(logits - m), axis=1, keepdims=True)
    expected_grad = probs / N
    for i in range(N):
        expected_grad[i, targets[i]] -= 1.0 / N

    loss, m_vec, l_vec = flash_cross_entropy_forward(logits, targets, chunk_size=16)
    grad = flash_cross_entropy_backward(logits, targets, m_vec, l_vec, chunk_size=16)

    assert np.allclose(grad, expected_grad, atol=1e-5)


def test_flash_cross_entropy_masked_leading_chunk():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((4, 150))
    logits[:, :64] = -np.inf
    targets = np.array([70, 80, 90, 100])
    row_max = logits.max(axis=1, keepdims=True)
    lse = row_max[:, 0] + np.log(np.exp(logits - row_max).sum(axis=1))
    expected = float(np.mean(lse - logits[np.arange(4), targets]))
    loss, _, _ = flash_cross_entropy_forward(logits, targets, chunk_size=64)
    assert np.isfinite(loss), loss
    assert abs(loss - expected) < 1e-12

