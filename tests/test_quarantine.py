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


def test_quarantine_thread_safety() -> None:
    import threading

    q = BackendQuarantine(failure_threshold=200)

    def worker() -> None:
        for _ in range(50):
            q.record_failure("gemm", "metal")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert q.failure_count("gemm", "metal") == 200
    assert q.is_quarantined("gemm", "metal")


def test_module_level_quarantine_helpers() -> None:
    from parallax.core.quarantine import (
        get_default_quarantine,
        is_quarantined,
        record_failure,
        reset_quarantine,
    )

    reset_quarantine()
    assert isinstance(get_default_quarantine(), BackendQuarantine)
    assert not is_quarantined("gemm", "metal")
    record_failure("gemm", "metal")
    record_failure("gemm", "metal")
    record_failure("gemm", "metal")
    assert is_quarantined("gemm", "metal")
    reset_quarantine()
    assert not is_quarantined("gemm", "metal")


def test_quarantine_all_exports() -> None:
    import parallax.core.quarantine as mod

    assert sorted(mod.__all__) == [
        "BackendQuarantine",
        "filter_backends",
        "get_default_quarantine",
        "is_quarantined",
        "record_failure",
        "record_success",
        "reset_quarantine",
    ]
    for sym in mod.__all__:
        assert hasattr(mod, sym)


def test_partial_reset_lifts_quarantine_after_success() -> None:
    from parallax.backends.quarantine import BackendQuarantine as BQ

    q = BQ(failure_threshold=2)
    q.record_failure("gemm", "c")
    q.record_failure("gemm", "c")
    assert q.is_quarantined("gemm", "c")
    q.record_success("gemm", "c")
    q.reset(op="gemm")
    assert not q.is_quarantined("gemm", "c")









