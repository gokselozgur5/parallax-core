from __future__ import annotations

from typing import Any
import numpy as np
import pytest

from parallax.backends.base import Backend
from parallax.backends.numpy_backend import NumpyBackend
from parallax.backends.parity import (
    ParityError,
    check_op_parity,
    check_parity,
    parity_errors,
)
from parallax.backends import registry


class MockBackend(Backend):
    def __init__(self, name: str, offset: float = 0.0, available: bool = True):
        self.name = name
        self.offset = offset
        self._available = available

    def available(self) -> bool:
        return self._available
    def supports(self, op: str, dtype: Any, shape: tuple[int, ...]) -> bool:
        return op in ("gemm", "softmax", "rmsnorm")

    def gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a @ b) + self.offset


def test_parity_exact_match() -> None:
    backend = MockBackend("exact", offset=0.0)
    a = np.ones((2, 2), dtype=np.float32)
    b = np.ones((2, 2), dtype=np.float32)
    report = check_parity("gemm", a, b, backends=[backend])
    assert "exact" in report
    res = report["exact"]
    assert res.passed
    assert res.status == "ok"
    assert res.max_abs_error == 0.0


def test_parity_threshold_flagging() -> None:
    ok_backend = MockBackend("slight", offset=1e-5)
    bad_backend = MockBackend("drift", offset=1e-3)
    a = np.eye(2, dtype=np.float32)
    b = np.eye(2, dtype=np.float32)
    report = check_parity("gemm", a, b, backends=[ok_backend, bad_backend])
    assert report["slight"].passed
    assert report["slight"].status == "ok"
    assert not report["drift"].passed
    assert report["drift"].status == "flagged"
    assert report.flagged_backends == ["drift"]
    assert not report.all_passed


def test_parity_unavailable_backend() -> None:
    backend = MockBackend("offline", available=False)
    a = np.ones((2, 2), dtype=np.float32)
    report = check_parity("gemm", a, a, backends=[backend])
    res = report["offline"]
    assert res.passed
    assert res.status == "unavailable"
    assert res.max_abs_error == 0.0


def test_parity_unsupported_op() -> None:
    backend = MockBackend("unsupported")
    backend.supports = lambda op, dt, shp: False  # type: ignore[method-assign]
    a = np.ones((2, 2), dtype=np.float32)
    report = check_parity("gemm", a, a, backends=[backend])
    res = report["unsupported"]
    assert res.passed
    assert res.status == "unsupported"


def test_parity_error_handling() -> None:
    backend = MockBackend("failing")
    backend.gemm = lambda a, b: 1 / 0  # type: ignore[method-assign]
    a = np.ones((2, 2), dtype=np.float32)
    report = check_parity("gemm", a, a, backends=[backend])
    assert not report["failing"].passed
    assert report["failing"].status == "error"
    assert "division by zero" in str(report["failing"].error_message)


def test_parity_raise_on_error() -> None:
    bad = MockBackend("bad", offset=1.0)
    a = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ParityError, match="parity check failed"):
        check_parity("gemm", a, a, backends=[bad], raise_on_error=True)
