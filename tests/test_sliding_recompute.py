from parallax.sliding_recompute import SlidingWindowBuffer


def test_sliding_window_buffer_init() -> None:
    buf = SlidingWindowBuffer(window_size=16, sink_size=4)
    assert buf.window_size == 16
    assert buf.sink_size == 4

import numpy as np


def test_filter_active_indices_bounds() -> None:
    buf = SlidingWindowBuffer(window_size=3, sink_size=2)
    # Total seq len = 10 -> sinks: 0, 1; window: 7, 8, 9
    indices = buf.filter_active_indices(total_seq_len=10)
    assert np.array_equal(indices, [0, 1, 7, 8, 9])
