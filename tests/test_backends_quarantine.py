from __future__ import annotations

import pytest
from parallax.backends.quarantine import BackendQuarantine


def test_quarantine_initial_state() -> None:
    q = BackendQuarantine(failure_threshold=3)
    assert q.threshold == 3
    assert not q.is_quarantined("gemm", "c")
    assert q.failure_count("gemm", "c") == 0


def test_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        BackendQuarantine(failure_threshold=0)


def test_consecutive_failures_trigger_quarantine() -> None:
    q = BackendQuarantine(failure_threshold=3)
    assert not q.record_failure("gemm", "c")
    assert q.failure_count("gemm", "c") == 1
    assert not q.is_quarantined("gemm", "c")

    assert not q.record_failure("gemm", "c")
    assert q.record_failure("gemm", "c")
    assert q.is_quarantined("gemm", "c")


def test_success_resets_failures() -> None:
    q = BackendQuarantine(failure_threshold=3)
    q.record_failure("gemm", "c")
    q.record_failure("gemm", "c")
    q.record_success("gemm", "c")
    assert q.failure_count("gemm", "c") == 0
    assert not q.is_quarantined("gemm", "c")


def test_filter_backends() -> None:
    q = BackendQuarantine(failure_threshold=2)
    q.record_failure("gemm", "c")
    q.record_failure("gemm", "c")
    assert q.filter_backends("gemm", ["c", "rust"]) == ["rust"]
    assert q.filter_backends("softmax", ["c", "rust"]) == ["c", "rust"]
