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
