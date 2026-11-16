"""Unit tests for precomputed RoPE tables and rotation dispatch helpers."""

import numpy as np
import pytest
from python.parallax.dispatch.rope import precompute_freqs


def test_precompute_freqs_shape_and_values() -> None:
    cos, sin = precompute_freqs(dim=16, max_seq_len=32, theta=10000.0)
    assert cos.shape == (32, 8)
    assert sin.shape == (32, 8)
    np.testing.assert_allclose(cos[0], np.ones(8, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(sin[0], np.zeros(8, dtype=np.float32), atol=1e-6)


def test_precompute_freqs_validation() -> None:
    with pytest.raises(ValueError):
        precompute_freqs(dim=15, max_seq_len=32)
    with pytest.raises(ValueError):
        precompute_freqs(dim=16, max_seq_len=0)
    with pytest.raises(ValueError):
        precompute_freqs(dim=16, max_seq_len=32, theta=-1.0)


def test_precompute_freqs_cis() -> None:
    from python.parallax.dispatch.rope import precompute_freqs_cis

    cis = precompute_freqs_cis(dim=8, max_seq_len=16, theta=10000.0)
    assert cis.shape == (16, 4)
    assert cis.dtype == np.complex64
    np.testing.assert_allclose(cis[0].real, np.ones(4), atol=1e-6)
    np.testing.assert_allclose(cis[0].imag, np.zeros(4), atol=1e-6)


def test_rotate_half() -> None:
    from python.parallax.dispatch.rope import rotate_half

    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out_non = rotate_half(x, interleaved=False)
    np.testing.assert_allclose(out_non, np.array([-3.0, -4.0, 1.0, 2.0]))

    out_int = rotate_half(x, interleaved=True)
    np.testing.assert_allclose(out_int, np.array([-2.0, 1.0, -4.0, 3.0]))


def test_apply_rotary_emb_identity_at_pos0() -> None:
    from python.parallax.dispatch.rope import apply_rotary_emb, precompute_freqs

    cos, sin = precompute_freqs(dim=4, max_seq_len=8)
    x = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
    out = apply_rotary_emb(x, cos, sin, offset=0, interleaved=False)
    np.testing.assert_allclose(out, x, atol=1e-6)

    out_int = apply_rotary_emb(x, cos, sin, offset=0, interleaved=True)
    np.testing.assert_allclose(out_int, x, atol=1e-6)
