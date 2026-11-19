"""Tests for core dispatch telemetry counters."""

from __future__ import annotations

import pytest

from parallax.core.telemetry import DispatchTelemetry


def test_record_and_op_counts():
    tel = DispatchTelemetry()
    assert tel.op_count("gemm") == 0
    assert tel.total_ops() == 0

    tel.record("gemm", "metal", 100)
    tel.record("gemm", "numpy", 200)
    tel.record("softmax", "c-neon", 50)

    assert tel.op_count("gemm") == 2
    assert tel.op_count("softmax") == 1
    assert tel.total_ops() == 3


def test_backend_counts_and_hit_rates():
    tel = DispatchTelemetry()
    assert tel.backend_count("metal") == 0
    assert tel.backend_hit_rate("metal") == 0.0

    tel.record("gemm", "metal", 100)
    tel.record("gemm", "metal", 100)
    tel.record("gemm", "metal", 100)
    tel.record("gemm", "numpy", 100)
    tel.record("softmax", "c-neon", 100)

    assert tel.backend_count("metal", "gemm") == 3
    assert tel.backend_count("numpy", "gemm") == 1
    assert tel.backend_hit_rate("metal", "gemm") == 0.75
    assert tel.backend_hit_rate("numpy", "gemm") == 0.25
    assert tel.backend_hit_rate("metal") == 0.6


def test_latency_counters_and_validation():
    tel = DispatchTelemetry()
    assert tel.total_latency_ns() == 0
    assert tel.average_latency_ns() == 0.0

    tel.record("gemm", "metal", 100)
    tel.record("gemm", "numpy", 200)
    tel.record("softmax", "c-neon", 300)

    assert tel.total_latency_ns("gemm") == 300
    assert tel.average_latency_ns("gemm") == 150.0
    assert tel.total_latency_ns() == 600
    assert tel.average_latency_ns() == 200.0

    with pytest.raises(ValueError, match="non-negative"):
        tel.record("gemm", "metal", -1)


def test_reset_and_concurrent_records():
    import threading

    tel = DispatchTelemetry()
    tel.record("gemm", "metal", 50)
    tel.reset()
    assert tel.total_ops() == 0
    assert tel.op_count("gemm") == 0
    assert tel.total_latency_ns() == 0

    threads = [
        threading.Thread(
            target=lambda: [tel.record("gemm", "metal", 10) for _ in range(500)]
        )
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert tel.total_ops() == 2000
    assert tel.op_count("gemm") == 2000
    assert tel.backend_count("metal", "gemm") == 2000
    assert tel.total_latency_ns("gemm") == 20000


def test_module_level_telemetry_helpers():
    from parallax.core.telemetry import (
        get_average_latency_ns,
        get_backend_hit_rate,
        get_op_count,
        get_telemetry,
        record,
        reset_telemetry,
    )

    reset_telemetry()
    assert isinstance(get_telemetry(), DispatchTelemetry)
    assert get_op_count("matmul") == 0

    record("matmul", "metal", 100)
    record("matmul", "metal", 200)
    record("matmul", "numpy", 300)

    assert get_op_count("matmul") == 3
    assert get_backend_hit_rate("metal", "matmul") == pytest.approx(2 / 3)
    assert get_average_latency_ns("matmul") == pytest.approx(200.0)

    reset_telemetry()
    assert get_op_count("matmul") == 0


def test_dispatch_records_into_backends_telemetry():
    from parallax.backends.telemetry import DispatchTelemetry as BackendTelemetry
    from parallax.core.telemetry import record, reset_telemetry

    reset_telemetry()
    record("gemm", "metal", 100)
    snap = BackendTelemetry().snapshot()
    assert snap[("gemm", "metal")].count == 1
    assert snap[("gemm", "metal")].total_ns == 100
    reset_telemetry()
