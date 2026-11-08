"""
Unit tests for StreamingLLM attention sink mask generation.
"""

import numpy as np
import pytest
from parallax.streaming import generate_sink_window_mask, streaming_attention


def test_sink_window_mask_structure():
    seq_len = 10
    window_size = 3
    num_sinks = 2
    mask = generate_sink_window_mask(seq_len, window_size=window_size, num_sinks=num_sinks)

    assert mask.shape == (seq_len, seq_len)
    # Upper triangle must strictly be False (causal)
    assert not np.any(np.triu(mask, k=1))
    # Sink column 0 must be True for all rows; column 1 must be True for all rows >= 1
    assert np.all(mask[:, 0])
    assert np.all(mask[1:, 1])


def test_streaming_attention_output_shape_and_finite():
    np.random.seed(42)
    B, H, T, d = 2, 2, 20, 8
    q, k, v = np.random.randn(B, H, T, d), np.random.randn(B, H, T, d), np.random.randn(B, H, T, d)

    out = streaming_attention(q, k, v, window_size=5, num_sinks=2)
    assert out.shape == (B, H, T, d)
    assert np.all(np.isfinite(out))


