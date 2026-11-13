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


def test_metal_stream_synchronize_and_context() -> None:
    s1 = MetalStream(stream_id=1, priority=1)
    s1.synchronize()
    orig = current_stream()
    with use_stream(s1) as active:
        assert active == s1
        assert current_stream() == s1
    assert current_stream() == orig
    s1.release()
    assert s1._queue is None


def test_metal_stream_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    stream = MetalStream(stream_id=9)
    assert not stream.available()
    assert stream._queue is None
    stream.synchronize()
    stream.release()
