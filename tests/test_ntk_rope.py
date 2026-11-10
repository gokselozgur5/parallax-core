import numpy as np
from parallax.attention import apply_rope
from parallax.ntk_rope import NTKAwareRoPE


def test_ntk_rope_initialization() -> None:
    rope = NTKAwareRoPE(dim=64, base=10000.0, alpha=2.0)
    assert rope.dim == 64
    assert rope.base == 10000.0
    assert rope.alpha == 2.0


def test_ntk_inv_freq_alpha_1_matches_standard() -> None:
    dim = 16
    base = 10000.0
    rope_standard = NTKAwareRoPE(dim=dim, base=base, alpha=1.0)
    rope_scaled = NTKAwareRoPE(dim=dim, base=base, alpha=4.0)

    std_freq = rope_standard.compute_scaled_inv_freq()
    scaled_freq = rope_scaled.compute_scaled_inv_freq()

    # Exact comparison against formula 1.0 / (base ** (indices / dim))
    indices = np.arange(0, dim, 2).astype(np.float64)
    expected_freq = 1.0 / (base ** (indices / dim))
    np.testing.assert_allclose(std_freq, expected_freq)

    # Cross check apply_rope with standard apply_rope for alpha=1.0
    x = np.random.randn(8, dim)
    out_ntk = rope_standard.apply_rope(x, seq_len=8)
    out_std = apply_rope(x[None, :, :], seq_dim=1, theta=base, interleaved=True)[0]
    np.testing.assert_allclose(out_ntk, out_std)

    # At index 0, base^0 == 1, so freq is 1. For higher indices, scaled_freq < std_freq
    assert scaled_freq[0] == std_freq[0]
    assert np.all(scaled_freq[1:] < std_freq[1:])


def test_apply_rope_preserves_vector_norm() -> None:
    np.random.seed(42)
    seq_len, dim = 8, 16
    rope = NTKAwareRoPE(dim=dim, alpha=2.0)
    x = np.random.randn(seq_len, dim)
    out = rope.apply_rope(x, seq_len=seq_len)
    assert out.shape == x.shape
    norm_before = np.linalg.norm(x, axis=-1)
    norm_after = np.linalg.norm(out, axis=-1)
    np.testing.assert_allclose(norm_before, norm_after, rtol=1e-5, atol=1e-5)
