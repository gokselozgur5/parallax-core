"""
Unit tests for StreamingLLM attention sink mask generation.
"""

import numpy as np
from parallax.streaming import (
    evict_kv_cache_with_sinks,
    generate_sink_window_mask,
    streaming_attention,
)


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


def test_evict_kv_cache_preserves_sinks_and_window():
    B, H, T, d = 1, 2, 30, 8
    num_sinks = 3
    window_size = 5
    k = np.arange(T)[None, None, :, None] * np.ones((B, H, T, d))
    v = k.copy()

    k_evicted, v_evicted = evict_kv_cache_with_sinks(
        k, v, window_size=window_size, num_sinks=num_sinks
    )
    # Target length: num_sinks + window_size = 3 + 5 = 8
    assert k_evicted.shape[2] == 8
    # Initial sinks must be exactly 0, 1, 2
    np.testing.assert_array_equal(k_evicted[0, 0, :3, 0], [0, 1, 2])
    # Rolling window must be the last 5 tokens (25, 26, 27, 28, 29)
    np.testing.assert_array_equal(k_evicted[0, 0, 3:, 0], [25, 26, 27, 28, 29])


def test_evict_kv_cache_window_size_zero():
    B, H, T, d = 1, 1, 5, 4
    num_sinks = 2
    window_size = 0
    k = np.arange(T)[None, None, :, None] * np.ones((B, H, T, d))
    v = k.copy()

    k_evicted, v_evicted = evict_kv_cache_with_sinks(
        k, v, window_size=window_size, num_sinks=num_sinks
    )
    assert k_evicted.shape[2] == num_sinks
    assert v_evicted.shape[2] == num_sinks
    np.testing.assert_array_equal(k_evicted[0, 0, :, 0], [0, 1])



