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
