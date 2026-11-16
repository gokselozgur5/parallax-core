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
