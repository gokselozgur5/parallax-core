"""Tests for NEON FlashAttention-1 single-head microkernel."""

import platform
import numpy as np
import pytest

if platform.machine() not in ("arm64", "aarch64"):
    pytest.skip("NEON FlashAttention requires arm64/aarch64", allow_module_level=True)

from parallax.native.flashattn_neon import (
    is_flashattn_neon_available,
    flashattn_neon,
)


def test_flashattn_neon_symbol_available():
    assert is_flashattn_neon_available()


def test_flashattn_neon_invalid_dims():
    q = np.ones((4, 16), dtype=np.float32)
    k = np.ones((4, 32), dtype=np.float32)
    v = np.ones((4, 32), dtype=np.float32)
    with pytest.raises(ValueError):
        flashattn_neon(q, k, v)


def test_flashattn_neon_parity_non_causal():
    rng = np.random.default_rng(42)
    seq_len, head_dim = 8, 32
    scale = 1.0 / np.sqrt(head_dim)
    q = rng.standard_normal((seq_len, head_dim)).astype(np.float32)
    k = rng.standard_normal((seq_len, head_dim)).astype(np.float32)
    v = rng.standard_normal((seq_len, head_dim)).astype(np.float32)
    scores = (q @ k.T) * scale
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    ref_attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    expected = ref_attn @ v
    actual = flashattn_neon(q, k, v, scale=scale, is_causal=False)
    assert np.allclose(actual, expected, atol=1e-4)


def test_flashattn_neon_parity_causal():
    rng = np.random.default_rng(123)
    seq_len, head_dim = 6, 24
    scale = 1.0 / np.sqrt(head_dim)
    q = rng.standard_normal((seq_len, head_dim)).astype(np.float32)
    k = rng.standard_normal((seq_len, head_dim)).astype(np.float32)
    v = rng.standard_normal((seq_len, head_dim)).astype(np.float32)
    scores = (q @ k.T) * scale
    mask = np.triu(np.full((seq_len, seq_len), -np.inf, dtype=np.float32), k=1)
    masked_scores = scores + mask
    exp_scores = np.exp(masked_scores - np.max(masked_scores, axis=-1, keepdims=True))
    ref_attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    expected = ref_attn @ v
    actual = flashattn_neon(q, k, v, scale=scale, is_causal=True)
    assert np.allclose(actual, expected, atol=1e-4)


def test_flashattn_neon_negative_inf_scores():
    seq_len, head_dim = 2, 8
    q = np.ones((seq_len, head_dim), dtype=np.float32)
    k = np.full((seq_len, head_dim), -np.inf, dtype=np.float32)
    v = np.ones((seq_len, head_dim), dtype=np.float32)
    res = flashattn_neon(q, k, v, scale=1.0, is_causal=False)
    assert not np.isnan(res).any()
