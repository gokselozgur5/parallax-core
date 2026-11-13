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
