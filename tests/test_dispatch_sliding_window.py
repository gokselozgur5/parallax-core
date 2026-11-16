"""Unit tests for sliding window FlashAttention dispatch helpers."""

import numpy as np
import pytest
from python.parallax.dispatch.sliding_window import build_sliding_window_mask


def test_build_sliding_window_mask_causal() -> None:
    mask = build_sliding_window_mask(seq_len_q=4, seq_len_k=4, window_size=1, causal=True)
    expected = np.array([
        [True, False, False, False],
        [True, True, False, False],
        [False, True, True, False],
        [False, False, True, True],
    ])
    np.testing.assert_array_equal(mask, expected)


def test_get_sliding_window_bounds() -> None:
    from python.parallax.dispatch.sliding_window import get_sliding_window_bounds

    starts, ends = get_sliding_window_bounds(seq_len_q=4, seq_len_k=4, window_size=2, causal=True)
    np.testing.assert_array_equal(starts, np.array([0, 0, 0, 1]))
    np.testing.assert_array_equal(ends, np.array([1, 2, 3, 4]))

    starts_nc, ends_nc = get_sliding_window_bounds(seq_len_q=4, seq_len_k=4, window_size=1, causal=False)
    np.testing.assert_array_equal(starts_nc, np.array([0, 0, 1, 2]))
    np.testing.assert_array_equal(ends_nc, np.array([2, 3, 4, 4]))
