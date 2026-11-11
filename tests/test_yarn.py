"""
Unit and numerical tests for YaRN context window extension (M25).
"""

import numpy as np

from parallax.attention import apply_rope
from parallax.yarn import apply_yarn_rope, yarn_find_correction_dim, yarn_linear_ramp_mask


def test_yarn_ramp_and_scale_boundaries():
    half_dim = 16
    mask = yarn_linear_ramp_mask(low=4.0, high=12.0, dim=half_dim)
    assert mask.shape == (half_dim,)
    assert mask[0] == 0.0
    assert mask[-1] == 1.0
    # Values between 4 and 12 must be strictly monotonic
    assert np.all(np.diff(mask[4:12]) >= 0.0)


def test_yarn_scale_factor_1_matches_standard_rope():
    np.random.seed(42)
    x = np.random.randn(2, 8, 4, 16)
    standard_out = apply_rope(x)
    # YaRN with scale_factor=1.0 should behave identically to standard RoPE
    yarn_out = apply_yarn_rope(x, scale_factor=1.0)
    np.testing.assert_allclose(yarn_out, standard_out, atol=1e-7)


def test_yarn_preserves_vector_norm_magnitude():
    x = np.random.randn(1, 16, 2, 8)
    norm_before = np.linalg.norm(x, axis=-1)
    yarn_out = apply_yarn_rope(x, scale_factor=4.0)
    # Check that norm scaling matches mscale factor
    expected_mscale = 0.1 * np.log(4.0) + 1.0
    norm_after = np.linalg.norm(yarn_out, axis=-1)
    np.testing.assert_allclose(norm_after / norm_before, expected_mscale, rtol=1e-5)


def test_yarn_frequency_interpolation_direction():
    half_dim = 64
    base = 10000.0
    max_pos = 2048
    low = yarn_find_correction_dim(32.0, half_dim, base, max_pos)
    high = yarn_find_correction_dim(1.0, half_dim, base, max_pos)
    ramp = yarn_linear_ramp_mask(low, high, half_dim)

    # High frequencies (d <= low): ramp must be 0 (no interpolation)
    assert np.all(ramp[:int(low)] == 0.0)
    # Low frequencies (d >= high): ramp must be 1 (full interpolation)
    assert np.all(ramp[int(np.ceil(high)):] == 1.0)
