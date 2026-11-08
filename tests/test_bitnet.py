"""
Unit tests for BitNet b1.58 ternary quantization (M30).
"""

import numpy as np
import pytest
from parallax.bitnet import quantize_weights_ternary


def test_quantize_weights_ternary_strictly_ternary():
    np.random.seed(42)
    w = np.random.randn(32, 64)
    w_ternary, gamma = quantize_weights_ternary(w)

    # Values must be strictly in {-1, 0, 1}
    unique_vals = set(np.unique(w_ternary))
    assert unique_vals.issubset({-1.0, 0.0, 1.0})
    assert gamma > 0.0
