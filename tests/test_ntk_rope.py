from parallax.ntk_rope import NTKAwareRoPE


def test_ntk_rope_initialization() -> None:
    rope = NTKAwareRoPE(dim=64, base=10000.0, alpha=2.0)
    assert rope.dim == 64
    assert rope.base == 10000.0
    assert rope.alpha == 2.0

def test_ntk_inv_freq_alpha_1_matches_standard() -> None:
    rope_standard = NTKAwareRoPE(dim=16, base=10000.0, alpha=1.0)
    rope_scaled = NTKAwareRoPE(dim=16, base=10000.0, alpha=4.0)
    std_freq = rope_standard.compute_scaled_inv_freq()
    scaled_freq = rope_scaled.compute_scaled_inv_freq()
    # At index 0, base^0 == 1, so freq is 1. For higher indices, scaled_freq < std_freq
    assert scaled_freq[0] == std_freq[0]
    assert np.all(scaled_freq[1:] < std_freq[1:])

import numpy as np


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
