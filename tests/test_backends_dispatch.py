from __future__ import annotations

from typing import Any
import numpy as np
import pytest

from parallax.backends import registry
from parallax.backends.base import Backend
from parallax.backends.dispatch import (
    FallbackRecord,
    clear_fallback_history,
    dispatch,
    get_fallback_history,
)


class MockBackend(Backend):
    def __init__(self, name: str, is_avail: bool = True, supports_op: bool = True) -> None:
        self.name = name
        self._avail = is_avail
        self._supports = supports_op

    def available(self) -> bool:
        return self._avail

    def supports(self, op: str, dtype: Any, shape: tuple[int, ...]) -> bool:
        return self._supports


@pytest.fixture(autouse=True)
def clean_registry_and_history():
    registry.clear()
    clear_fallback_history()
    yield
    registry.clear()
    clear_fallback_history()


def test_fallback_record_repr_and_clearing():
    record = FallbackRecord(op="gemm", backend="mock", reason="failed", exception=None)
    assert record.op == "gemm"
    assert record.backend == "mock"
    assert record.reason == "failed"
    assert get_fallback_history() == []
    clear_fallback_history()
    assert get_fallback_history() == []


class CalculatorBackend(MockBackend):
    def gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b


def test_dispatch_success_on_first_candidate():
    backend = CalculatorBackend("numpy", is_avail=True, supports_op=True)
    registry.register("numpy", backend)
    a = np.ones((2, 2), dtype=np.float32)
    res = dispatch("gemm", a, a)
    np.testing.assert_array_equal(res, np.full((2, 2), 2.0, dtype=np.float32))
    assert get_fallback_history() == []


def test_dispatch_skips_unsupported_backend():
    unsupported = CalculatorBackend("c", is_avail=True, supports_op=False)
    supported = CalculatorBackend("numpy", is_avail=True, supports_op=True)
    registry.register("c", unsupported)
    registry.register("numpy", supported)
    a = np.ones((2, 2), dtype=np.float32)
    res = dispatch("gemm", a, a, _environ={"PARALLAX_BACKEND": "c"})
    assert res is not None
    assert len(get_fallback_history()) == 1


class FaultyBackend(MockBackend):
    def gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        raise RuntimeError("Kernel crashed")


def test_dispatch_falls_back_on_execution_exception():
    faulty = FaultyBackend("c", is_avail=True, supports_op=True)
    fallback = CalculatorBackend("numpy", is_avail=True, supports_op=True)
    registry.register("c", faulty)
    registry.register("numpy", fallback)
    a = np.ones((2, 2), dtype=np.float32)
    res = dispatch("gemm", a, a, _environ={"PARALLAX_BACKEND": "c"})
    assert res is not None
    history = get_fallback_history()
    assert len(history) == 1
    assert "Kernel crashed" in history[0].reason


def test_dispatch_raises_when_all_fail():
    faulty = FaultyBackend("numpy", is_avail=True, supports_op=True)
    registry.register("numpy", faulty)
    with pytest.raises(RuntimeError, match="All backends failed"):
        dispatch("gemm", np.ones((2, 2), dtype=np.float32), np.ones((2, 2)))
