"""Unit tests for precomputed RoPE tables and rotation dispatch helpers."""

import numpy as np
import pytest
from parallax.dispatch.rope import precompute_freqs


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
    from parallax.dispatch.rope import precompute_freqs_cis

    cis = precompute_freqs_cis(dim=8, max_seq_len=16, theta=10000.0)
    assert cis.shape == (16, 4)
    assert cis.dtype == np.complex64
    np.testing.assert_allclose(cis[0].real, np.ones(4), atol=1e-6)
    np.testing.assert_allclose(cis[0].imag, np.zeros(4), atol=1e-6)


def test_rotate_half() -> None:
    from parallax.dispatch.rope import rotate_half

    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out_non = rotate_half(x, interleaved=False)
    np.testing.assert_allclose(out_non, np.array([-3.0, -4.0, 1.0, 2.0]))

    out_int = rotate_half(x, interleaved=True)
    np.testing.assert_allclose(out_int, np.array([-2.0, 1.0, -4.0, 3.0]))


def test_apply_rotary_emb_identity_at_pos0() -> None:
    from parallax.dispatch.rope import apply_rotary_emb, precompute_freqs

    cos, sin = precompute_freqs(dim=4, max_seq_len=8)
    x = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
    out = apply_rotary_emb(x, cos, sin, offset=0, interleaved=False)
    np.testing.assert_allclose(out, x, atol=1e-6)

    out_int = apply_rotary_emb(x, cos, sin, offset=0, interleaved=True)
    np.testing.assert_allclose(out_int, x, atol=1e-6)


def test_apply_rotary_emb_parity_with_reference() -> None:
    from parallax.attention import apply_rope
    from parallax.dispatch.rope import apply_rotary_emb, precompute_freqs

    rng = np.random.default_rng(42)
    x = rng.standard_normal((8, 2, 16)).astype(np.float32)
    cos, sin = precompute_freqs(dim=16, max_seq_len=32, theta=10000.0)
    act_int = apply_rotary_emb(x, cos, sin, offset=3, interleaved=True)
    x_t = np.transpose(x, (1, 0, 2))
    exp_int = np.transpose(apply_rope(x_t, seq_dim=1, offset=3, interleaved=True), (1, 0, 2))
    np.testing.assert_allclose(act_int, exp_int, atol=1e-5, rtol=1e-5)
    act_non = apply_rotary_emb(x, cos, sin, offset=3, interleaved=False)
    exp_non = np.transpose(apply_rope(x_t, seq_dim=1, offset=3, interleaved=False), (1, 0, 2))
    np.testing.assert_allclose(act_non, exp_non, atol=1e-5, rtol=1e-5)


def test_rope_frequency_table_and_slice() -> None:
    from parallax.dispatch.rope import RoPEFrequencyTable

    table = RoPEFrequencyTable(dim=16, max_seq_len=64)
    c, s = table.get_slice(seq_len=8, offset=4)
    assert c.shape == (8, 8)
    assert s.shape == (8, 8)
    x = np.ones((8, 2, 16), dtype=np.float32)
    out = table.rotate(x, offset=4, interleaved=True)
    assert out.shape == x.shape
    with pytest.raises(IndexError):
        table.get_slice(seq_len=60, offset=10)


def test_dispatch_rope() -> None:
    from parallax.dispatch.rope import dispatch_rope

    x = np.ones((4, 2, 8), dtype=np.float32)
    out = dispatch_rope(x, start_pos=0, interleaved=True)
    assert out.shape == x.shape
    assert out.dtype == np.float32


def test_package_exports() -> None:
    import parallax.dispatch as pd
    import parallax.dispatch as ppd

    for mod in (pd, ppd):
        assert hasattr(mod, "RoPEFrequencyTable")
        assert hasattr(mod, "apply_rotary_emb")
        assert hasattr(mod, "dispatch_rope")
        assert hasattr(mod, "precompute_freqs")
        assert hasattr(mod, "precompute_freqs_cis")
        assert hasattr(mod, "rotate_half")
