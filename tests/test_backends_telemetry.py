import pytest

from parallax.backends.telemetry import CallStats, DispatchTelemetry


def test_record_accumulates_per_op_and_backend():
    tel = DispatchTelemetry()
    tel.record("gemm", "metal", 100)
    tel.record("gemm", "metal", 50)
    tel.record("gemm", "numpy", 7)
    snap = tel.snapshot()
    assert snap[("gemm", "metal")] == CallStats(count=2, total_ns=150)
    assert snap[("gemm", "numpy")] == CallStats(count=1, total_ns=7)


def test_snapshot_is_read_only_and_detached():
    tel = DispatchTelemetry()
    tel.record("softmax", "rust", 1)
    snap = tel.snapshot()
    with pytest.raises(TypeError):
        snap[("softmax", "rust")] = CallStats()  # type: ignore[index]
    tel.record("softmax", "rust", 1)
    assert snap[("softmax", "rust")].count == 1
