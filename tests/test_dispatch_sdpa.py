"""Tests for scaled dot-product attention (SDPA) dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from python.parallax.dispatch.sdpa import (
    DispatchedSDPA,
    SDPAModule,
    dispatched_sdpa,
    scaled_dot_product_attention,
    sdpa_backward,
)


def test_sdpa_forward_shape_and_dtype() -> None:
    q = np.random.randn(2, 4, 8, 16).astype(np.float32)
    k = np.random.randn(2, 4, 8, 16).astype(np.float32)
    v = np.random.randn(2, 4, 8, 16).astype(np.float32)
    out = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 4, 8, 16)
    assert out.dtype == np.float32


def test_sdpa_invalid_dims_raise() -> None:
    q = np.ones((4, 8))
    k_bad = np.ones((4, 12))
    v = np.ones((4, 8))
    with pytest.raises(ValueError, match="query head_dim"):
        scaled_dot_product_attention(q, k_bad, v)
    k_bad_len = np.ones((5, 8))
    v_bad_len = np.ones((4, 8))
    with pytest.raises(ValueError, match="key seq_len"):
        scaled_dot_product_attention(q, k_bad_len, v_bad_len)
    with pytest.raises(ValueError, match="at least 2D"):
        scaled_dot_product_attention(np.ones(4), np.ones(4), np.ones(4))
