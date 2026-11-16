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
