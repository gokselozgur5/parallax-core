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


def test_parity_checker_unavailable_and_unsupported():
    from parallax.core.parity import ParityChecker

    class Unavail(MockBackend):
        def available(self) -> bool:
            return False

    class Unsupp(MockBackend):
        def supports(self, op: str, dtype: object = None, shape: tuple[int, ...] = ()) -> bool:
            return False

    checker = ParityChecker(reference=MockBackend("ref"))
    a = np.ones((2, 2), dtype=np.float32)
    report = checker.check_parity("gemm", a, a, backends=[Unavail("off"), Unsupp("unsupp")])
    assert report["off"].status == "unavailable" and report["off"].passed is True
    assert report["unsupp"].status == "unsupported" and report["unsupp"].passed is True


def test_parity_module_helpers_and_raise_on_error():
    from parallax.core.parity import check_parity, parity_errors

    bad = MockBackend("drift", offset=1.0)
    ref = MockBackend("ref", offset=0.0)
    reg = {"drift": bad, "ref": ref}
    a = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ParityError, match="parity check failed"):
        check_parity("gemm", a, a, backends=["drift"], reference="ref", registry=reg, raise_on_error=True)

    errs = parity_errors("gemm", a, a, backends=["ref"], reference="ref", registry=reg)
    assert "ref" in errs and errs["ref"] == 0.0


def test_parity_real_numpy_backend():
    from parallax.backends.numpy_backend import NumpyBackend
    from parallax.core.parity import check_parity

    nb = NumpyBackend()
    x = np.random.randn(4, 8).astype(np.float32)
    w = np.ones((8,), dtype=np.float32)
    report = check_parity("rmsnorm", x, w, backends=[nb], reference=nb)
    assert report["numpy"].passed is True
    assert report["numpy"].max_abs_error == 0.0


def test_parity_errors_includes_flagged_backends():
    from parallax.backends import registry
    from parallax.backends.base import Backend
    from parallax.backends.parity import parity_errors

    class MockBackendB(MockBackend, Backend):
        pass

    drift = MockBackendB("drift", offset=1e-3)
    registry.register("drift", drift)
    try:
        a = np.eye(2, dtype=np.float32)
        errs = parity_errors("gemm", a, a, backends=["drift"], reference=MockBackendB("ref"))
        assert "drift" in errs
        assert abs(errs["drift"] - 1e-3) < 1e-6
    finally:
        registry.unregister("drift")






