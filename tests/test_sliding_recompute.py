from parallax.sliding_recompute import SlidingWindowBuffer


def test_sliding_window_buffer_init() -> None:
    buf = SlidingWindowBuffer(window_size=16, sink_size=4)
    assert buf.window_size == 16
    assert buf.sink_size == 4
