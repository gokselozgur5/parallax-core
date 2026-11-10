import numpy as np
from parallax.ring_attention import RingAttentionContext


def test_ring_attention_context_init() -> None:
    ctx = RingAttentionContext(block_size=32)
    assert ctx.block_size == 32

from parallax.ring_attention import ring_flash_update


def test_ring_flash_update_rescale() -> None:
    acc_o = np.ones((2, 4))
    acc_m = np.zeros(2)
    acc_l = np.ones(2)
    block_o = np.full((2, 4), 2.0)
    block_m = np.ones(2)
    block_l = np.ones(2)
    merged_o, merged_m, merged_l = ring_flash_update(
        acc_o, acc_m, acc_l, block_o, block_m, block_l
    )
    assert merged_o.shape == (2, 4)
    assert np.all(merged_m == 1.0)
    # Analytical expected output: (1.0 * (1/e) + 2.0 * 1.0) / (1/e + 1.0)
    expected_val = (1.0 / np.e + 2.0) / (1.0 / np.e + 1.0)
    np.testing.assert_allclose(merged_o, expected_val, rtol=1e-6, atol=1e-6)

from parallax.ring_attention import ring_attention_forward


def test_ring_attention_matches_standard_causal_attention() -> None:
    np.random.seed(42)
    seq_len, dim = 32, 16
    q = np.random.randn(seq_len, dim)
    k = np.random.randn(seq_len, dim)
    v = np.random.randn(seq_len, dim)

    # Reference full causal attention
    scale = 1.0 / np.sqrt(dim)
    scores = np.matmul(q, k.T) * scale
    mask = np.tril(np.ones((seq_len, seq_len)))
    scores = np.where(mask == 1, scores, -1e9)
    exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    ref_out = np.matmul(weights, v)

    ring_out = ring_attention_forward(q, k, v, block_size=8, causal=True)
    np.testing.assert_allclose(ring_out, ref_out, rtol=1e-5, atol=1e-5)


def _naive_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool = False) -> np.ndarray:
    Tq, dim = q.shape
    Tk = k.shape[0]
    scale = 1.0 / np.sqrt(dim)
    scores = np.matmul(q, k.T) * scale
    if causal:
        mask = np.arange(Tq)[:, None] >= np.arange(Tk)[None, :]
        scores = np.where(mask, scores, -1e9)
    exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    return np.matmul(weights, v)


def test_ring_attention_asymmetric_tq_tk_non_causal_and_causal() -> None:
    np.random.seed(42)
    dim = 8
    # Test specific reproduction cases: Tq=3, Tk=7 and Tq=7, Tk=3
    for Tq, Tk in [(3, 7), (7, 3), (11, 5), (4, 18)]:
        for causal in [False, True]:
            for block_size in [2, 4, 8]:
                q = np.random.randn(Tq, dim)
                k = np.random.randn(Tk, dim)
                v = np.random.randn(Tk, dim)

                ref_out = _naive_attention(q, k, v, causal=causal)
                ring_out = ring_attention_forward(
                    q, k, v, block_size=block_size, causal=causal
                )
                np.testing.assert_allclose(ring_out, ref_out, rtol=1e-5, atol=1e-5)


def test_ring_attention_context_forward_and_validation() -> None:
    np.random.seed(123)
    ctx = RingAttentionContext(block_size=4)
    q = np.random.randn(6, 8)
    k = np.random.randn(10, 8)
    v = np.random.randn(10, 8)

    out_ctx = ctx.forward(q, k, v, causal=False)
    out_direct = ring_attention_forward(q, k, v, block_size=4, causal=False)
    np.testing.assert_array_equal(out_ctx, out_direct)

