import numpy as np
import pytest
from parallax.native.flashattn_avx512 import (
    _get_flashattn_avx512_fn,
    is_avx512_flashattn_available,
    flashattn_avx512,
)


def test_flashattn_avx512_symbols_bound():
    fn = _get_flashattn_avx512_fn()
    assert fn is not None
    assert isinstance(is_avx512_flashattn_available(), bool)


def test_flashattn_avx512_dim_and_type_checks():
    q = np.ones((2, 2), dtype=np.float32)
    k = np.ones((2, 3), dtype=np.float32)
    v = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        flashattn_avx512(q, k, v)


def test_flashattn_avx512_execution_or_safe_guard():
    seq_len, head_dim = 4, 16
    q = np.random.randn(seq_len, head_dim).astype(np.float32)
    k = np.random.randn(seq_len, head_dim).astype(np.float32)
    v = np.random.randn(seq_len, head_dim).astype(np.float32)
    if is_avx512_flashattn_available():
        out = flashattn_avx512(q, k, v)
        assert out.shape == (seq_len, head_dim)
    else:
        with pytest.raises(RuntimeError, match="AVX-512F is not supported on this CPU"):
            flashattn_avx512(q, k, v)


def test_flashattn_avx512_parity_skipped_when_unavailable():
    if not is_avx512_flashattn_available():
        pytest.skip("AVX-512F ISA not supported on this host")
    seq_len, head_dim = 8, 32
    scale = 1.0 / np.sqrt(head_dim)
    q = np.random.randn(seq_len, head_dim).astype(np.float32)
    k = np.random.randn(seq_len, head_dim).astype(np.float32)
    v = np.random.randn(seq_len, head_dim).astype(np.float32)
    res = flashattn_avx512(q, k, v, scale=scale, is_causal=False)
    scores = (q @ k.T) * scale
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    ref_attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    expected = ref_attn @ v
    assert np.allclose(res, expected, atol=1e-4)
