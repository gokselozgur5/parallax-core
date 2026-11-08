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
