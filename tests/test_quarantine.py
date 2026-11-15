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
