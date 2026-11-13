from __future__ import annotations

import sys
import pytest
from parallax.backends.metal_stream import (
    MetalStream,
    current_stream,
    use_stream,
)
from parallax.metal_engine import is_metal_available


def test_metal_stream_availability() -> None:
    stream = MetalStream(stream_id=1)
    if sys.platform != "darwin":
        assert not stream.available()
        assert stream._queue is None
    else:
        assert stream.available() == is_metal_available()
        if stream.available():
            assert stream._queue is not None
    stream.release()
