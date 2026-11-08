from parallax.chunked_prefill import make_chunked_causal_mask
import numpy as np


def test_make_chunked_causal_mask_structure():
    chunk_len = 3
    prefix_len = 2
    mask = make_chunked_causal_mask(chunk_len=chunk_len, prefix_len=prefix_len)

    assert mask.shape == (3, 5)
    # First 2 columns (prefix) must be fully True for all chunk rows
    assert np.all(mask[:, :2])
    # Intra-chunk causality: row 0 attends to col 2, but not 3, 4
    assert mask[0, 2] and not mask[0, 3] and not mask[0, 4]
    # Row 1 attends to col 2, 3, but not 4
    assert mask[1, 2] and mask[1, 3] and not mask[1, 4]
    # Row 2 attends to all cols 2, 3, 4
    assert mask[2, 2] and mask[2, 3] and mask[2, 4]


from parallax.chunked_prefill import chunked_prefill_attention


def test_chunked_prefill_attention_matches_full_causal_attention():
    np.random.seed(42)
    total_len = 16
    d = 8
    q_full = np.random.randn(total_len, d).astype(np.float32)
    k_full = np.random.randn(total_len, d).astype(np.float32)
    v_full = np.random.randn(total_len, d).astype(np.float32)

    # Reference full causal attention
    scale = 1.0 / np.sqrt(d)
    logits = np.dot(q_full, k_full.T) * scale
    mask = np.triu(np.ones((total_len, total_len), dtype=bool), k=1)
    logits[mask] = -1e9
    m = np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits - m) / np.sum(np.exp(logits - m), axis=1, keepdims=True)
    expected_out = np.dot(weights, v_full)

    # Chunked prefill in 2 chunks of size 8
    out_chunk0 = chunked_prefill_attention(
        q_chunk=q_full[:8],
        k_all=k_full[:8],
        v_all=v_full[:8],
        prefix_len=0,
    )
    out_chunk1 = chunked_prefill_attention(
        q_chunk=q_full[8:],
        k_all=k_full[:16],
        v_all=v_full[:16],
        prefix_len=8,
    )
    reconstructed_out = np.concatenate([out_chunk0, out_chunk1], axis=0)

    assert np.allclose(reconstructed_out, expected_out, atol=1e-5)
