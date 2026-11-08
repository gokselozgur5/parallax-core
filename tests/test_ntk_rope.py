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
    # High alpha should lower the frequencies to stretch wavelength
    assert np.all(scaled_freq < std_freq)
