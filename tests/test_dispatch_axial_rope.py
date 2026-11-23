"""Tests for 2D Axial RoPE dispatch helpers for multimodal and vision models."""

import numpy as np
import pytest
from parallax.dispatch.axial_rope import (
    AxialRoPEFrequencyTable,
    apply_axial_rope,
    apply_axial_rope_grid,
    dispatch_axial_rope,
    precompute_axial_freqs,
    precompute_axial_freqs_cis,
)


def test_precompute_axial_freqs_shapes():
    cos_h, sin_h, cos_w, sin_w = precompute_axial_freqs(dim=64, max_h=16, max_w=16)
    assert cos_h.shape == (16, 16)
    assert sin_h.shape == (16, 16)
    assert cos_w.shape == (16, 16)
    assert sin_w.shape == (16, 16)


def test_precompute_axial_freqs_validation():
    with pytest.raises(ValueError, match="positive and divisible by 4"):
        precompute_axial_freqs(dim=18, max_h=8, max_w=8)
    with pytest.raises(ValueError, match="max_h and max_w must be positive"):
        precompute_axial_freqs(dim=32, max_h=0, max_w=8)


def test_precompute_axial_freqs_cis():
    cis_h, cis_w = precompute_axial_freqs_cis(dim=32, max_h=8, max_w=8)
    assert cis_h.shape == (8, 8)
    assert cis_w.shape == (8, 8)
    assert cis_h.dtype == np.complex64
    assert cis_w.dtype == np.complex64


def test_apply_axial_rope_norm_preservation():
    cos_h, sin_h, cos_w, sin_w = precompute_axial_freqs(dim=32, max_h=8, max_w=8)
    x = np.random.randn(2, 4, 32).astype(np.float32)
    h_coords = np.array([2, 5], dtype=np.int64)
    w_coords = np.array([3, 1], dtype=np.int64)
    out = apply_axial_rope(x, cos_h, sin_h, cos_w, sin_w, h_coords, w_coords)
    assert out.shape == x.shape
    np.testing.assert_allclose(np.linalg.norm(out, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-5)


def test_apply_axial_rope_known_rotation_angle():
    cos_h, sin_h, cos_w, sin_w = precompute_axial_freqs(dim=4, max_h=4, max_w=4, theta=100.0)
    x = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    h_coords = np.array([1], dtype=np.int64)
    w_coords = np.array([2], dtype=np.int64)
    out = apply_axial_rope(x, cos_h, sin_h, cos_w, sin_w, h_coords, w_coords)
    expected = np.array(
        [[cos_h[1, 0], sin_h[1, 0], -sin_w[2, 0], cos_w[2, 0]]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out, expected, rtol=1e-5)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(x, expected, rtol=1e-5)


def test_apply_axial_rope_grid_patches():
    cos_h, sin_h, cos_w, sin_w = precompute_axial_freqs(dim=32, max_h=8, max_w=8)
    x = np.random.randn(16, 32).astype(np.float32)
    out = apply_axial_rope_grid(x, cos_h, sin_h, cos_w, sin_w, height=4, width=4)
    assert out.shape == (16, 32)
    np.testing.assert_allclose(out[0], x[0], rtol=1e-5)


def test_axial_rope_frequency_table_container():
    table = AxialRoPEFrequencyTable(dim=32, max_h=16, max_w=16)
    assert table.dim == 32
    assert table.max_h == 16
    assert table.max_w == 16
    x = np.random.randn(9, 32).astype(np.float32)
    out = table.apply_to_grid(x, height=3, width=3)
    assert out.shape == (9, 32)


def test_dispatch_axial_rope_execution():
    x = np.random.randn(6, 32).astype(np.float32)
    out = dispatch_axial_rope(x, height=2, width=3)
    assert out.shape == (6, 32)


def test_axial_rope_package_exports():
    import parallax.dispatch as pd
    import parallax.dispatch as ppd

    symbols = [
        "AxialRoPEFrequencyTable",
        "apply_axial_rope",
        "apply_axial_rope_grid",
        "dispatch_axial_rope",
        "precompute_axial_freqs",
        "precompute_axial_freqs_cis",
    ]
    for mod in (pd, ppd):
        for sym in symbols:
            assert hasattr(mod, sym)
