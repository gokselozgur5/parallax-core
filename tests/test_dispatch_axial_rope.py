"""Tests for 2D Axial RoPE dispatch helpers for multimodal and vision models."""

import numpy as np
import pytest
from python.parallax.dispatch.axial_rope import precompute_axial_freqs


def test_precompute_axial_freqs_shapes():
    cos_h, sin_h, cos_w, sin_w = precompute_axial_freqs(dim=64, max_h=16, max_w=16)
    assert cos_h.shape == (16, 16)
    assert sin_h.shape == (16, 16)
    assert cos_w.shape == (16, 16)
    assert sin_w.shape == (16, 16)


def test_precompute_axial_freqs_validation():
    with pytest.raises(ValueError, match="dim must be divisible by 4"):
        precompute_axial_freqs(dim=18, max_h=8, max_w=8)
    with pytest.raises(ValueError, match="max_h and max_w must be positive"):
        precompute_axial_freqs(dim=32, max_h=0, max_w=8)
