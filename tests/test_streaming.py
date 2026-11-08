"""
Unit tests for StreamingLLM attention sink mask generation.
"""

import numpy as np
import pytest
from parallax.streaming import generate_sink_window_mask


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

