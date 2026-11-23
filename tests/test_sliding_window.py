"""Unit tests for sliding window attention causal offset handling."""

import numpy as np
from parallax.dispatch.sliding_window import (
    build_sliding_window_mask,
    get_sliding_window_bounds,
)


def test_causal_sliding_window_mask_offset() -> None:
    mask = build_sliding_window_mask(seq_len_q=1, seq_len_k=10, window_size=3, causal=True)
    expected = np.array([[False, False, False, False, False, False, True, True, True, True]])
    np.testing.assert_array_equal(mask, expected)
    starts, ends = get_sliding_window_bounds(seq_len_q=1, seq_len_k=10, window_size=3, causal=True)
    np.testing.assert_array_equal(starts, np.array([6]))
    np.testing.assert_array_equal(ends, np.array([10]))
