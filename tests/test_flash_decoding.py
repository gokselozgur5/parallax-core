import numpy as np
from parallax.flash_decoding import flash_decode_split_kernel


def test_flash_decode_split_kernel_math():
    np.random.seed(42)
    d = 16
    chunk_len = 8
    q = np.random.randn(d)
    k_chunk = np.random.randn(chunk_len, d)
    v_chunk = np.random.randn(chunk_len, d)
    scale = 1.0 / np.sqrt(d)

    out, m_val, l_val = flash_decode_split_kernel(q, k_chunk, v_chunk, scale)

    logits = np.dot(k_chunk, q) * scale
    assert np.isclose(m_val, np.max(logits))
    assert np.isclose(l_val, np.sum(np.exp(logits - m_val)))
    expected_out = np.dot(np.exp(logits - m_val), v_chunk)
    assert np.allclose(out, expected_out)


from parallax.flash_decoding import reduce_split_k_outputs


def test_reduce_split_k_outputs_exact_match():
    np.random.seed(42)
    num_splits = 4
    d = 16
    local_outs = np.random.randn(num_splits, d)
    local_maxes = np.array([2.5, 3.1, 1.8, 3.0])
    local_lsums = np.array([12.0, 18.5, 9.2, 14.1])

    out = reduce_split_k_outputs(local_outs, local_maxes, local_lsums)
    assert out.shape == (d,)
    assert not np.isnan(out).any()


from parallax.flash_decoding import flash_decode_attention


def test_flash_decode_attention_matches_standard_dot_product():
    np.random.seed(42)
    seq_len = 256
    d = 32
    q = np.random.randn(d)
    k = np.random.randn(seq_len, d)
    v = np.random.randn(seq_len, d)

    # Standard attention reference
    scale = 1.0 / np.sqrt(d)
    logits = np.dot(k, q) * scale
    max_logit = np.max(logits)
    weights = np.exp(logits - max_logit)
    weights /= np.sum(weights)
    expected_out = np.dot(weights, v)

    # FlashDecoding output with split_size=64 (4 splits)
    fd_out = flash_decode_attention(q, k, v, split_size=64)

    assert np.allclose(fd_out, expected_out, atol=1e-6)
