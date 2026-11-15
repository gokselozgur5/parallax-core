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
