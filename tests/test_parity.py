"""Tests for cross-backend numerical parity checker."""

from __future__ import annotations

import numpy as np
import pytest

from parallax.core.parity import (
    ParityError,
    ParityReport,
    ParityResult,
)


def test_parity_result_and_report_contract():
    res = ParityResult("mock", "gemm", 0.0, True, "ok")
    assert res.backend == "mock"
    assert res.op == "gemm"
    assert res.passed is True
    assert res.status == "ok"
    report = ParityReport({"mock": res})
    assert report.all_passed is True
    assert report.flagged_backends == []
    assert report.max_errors == {"mock": 0.0}


class MockBackend:
    def __init__(self, name: str, offset: float = 0.0, avail: bool = True):
        self.name, self.offset, self._avail = name, offset, avail

    def available(self) -> bool:
        return self._avail

    def supports(self, op: str, dtype: object = None, shape: tuple[int, ...] = ()) -> bool:
        return True

    def gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a @ b) + self.offset


def test_parity_checker_exact_and_flagged():
    from parallax.core.parity import ParityChecker

    be_ok, be_bad = MockBackend("slight", offset=1e-5), MockBackend("bad", offset=1e-2)
    checker = ParityChecker(reference=MockBackend("ref"), registry={"slight": be_ok, "bad": be_bad})
    a = np.eye(2, dtype=np.float32)
    report = checker.check_parity("gemm", a, a, backends=["slight", "bad"])
    assert report["slight"].passed is True
    assert report["bad"].passed is False
    assert report.flagged_backends == ["bad"]


