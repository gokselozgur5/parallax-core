"""Tests for backend quarantine in parallax.core."""

from __future__ import annotations

import pytest

from parallax.core.quarantine import BackendQuarantine


def test_quarantine_initial_state_and_threshold() -> None:
    q = BackendQuarantine(failure_threshold=3)
    assert q.threshold == 3
    assert not q.is_quarantined("gemm", "metal")
    assert q.failure_count("gemm", "metal") == 0


def test_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be >= 1"):
        BackendQuarantine(failure_threshold=0)


def test_consecutive_failures_trigger_quarantine() -> None:
    q = BackendQuarantine(failure_threshold=3)
    assert not q.record_failure("gemm", "metal")
    assert q.failure_count("gemm", "metal") == 1
    assert not q.is_quarantined("gemm", "metal")
    assert not q.record_failure("gemm", "metal")
    assert q.record_failure("gemm", "metal")
    assert q.is_quarantined("gemm", "metal")


def test_success_resets_failures() -> None:
    q = BackendQuarantine(failure_threshold=3)
    q.record_failure("gemm", "metal")
    q.record_failure("gemm", "metal")
    assert q.failure_count("gemm", "metal") == 2
    q.record_success("gemm", "metal")
    assert q.failure_count("gemm", "metal") == 0
    assert not q.is_quarantined("gemm", "metal")


def test_filter_backends() -> None:
    q = BackendQuarantine(failure_threshold=2)
    q.record_failure("gemm", "metal")
    q.record_failure("gemm", "metal")
    assert q.filter_backends("gemm", ["metal", "numpy", "rust"]) == ["numpy", "rust"]
    assert q.filter_backends("softmax", ["metal", "numpy"]) == ["metal", "numpy"]


def test_quarantine_reset() -> None:
    q = BackendQuarantine(failure_threshold=2)
    q.record_failure("gemm", "metal")
    q.record_failure("gemm", "metal")
    assert q.is_quarantined("gemm", "metal")
    q.reset(op="gemm", backend="metal")
    assert not q.is_quarantined("gemm", "metal")
    q.record_failure("gemm", "metal")
    q.record_failure("gemm", "metal")
    q.reset()
    assert not q.is_quarantined("gemm", "metal")


def test_quarantine_cooldown_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 100.0
    monkeypatch.setattr("time.monotonic", lambda: now)
    q = BackendQuarantine(failure_threshold=2, cooldown_seconds=10.0)
    assert q.cooldown == 10.0
    q.record_failure("gemm", "metal")
    q.record_failure("gemm", "metal")
    assert q.is_quarantined("gemm", "metal")

    now = 105.0
    assert q.is_quarantined("gemm", "metal")

    now = 111.0
    assert not q.is_quarantined("gemm", "metal")





