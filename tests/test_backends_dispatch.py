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


from parallax.backends.dispatch import set_scope_backend


def test_dispatch_scope_override_precedence():
    backend = CalculatorBackend("rust", is_avail=True, supports_op=True)
    registry.register("rust", backend)
    set_scope_backend("rust")
    try:
        a = np.ones((2, 2), dtype=np.float32)
        res = dispatch("gemm", a, a)
        assert res is not None
        assert get_fallback_history() == []
    finally:
        set_scope_backend(None)


def test_dispatch_unavailable_backend_fallback():
    unavail = CalculatorBackend("rust", is_avail=False, supports_op=True)
    fallback = CalculatorBackend("numpy", is_avail=True, supports_op=True)
    registry.register("rust", unavail)
    registry.register("numpy", fallback)
    a = np.ones((2, 2), dtype=np.float32)
    res = dispatch("gemm", a, a, _environ={"PARALLAX_BACKEND": "rust"})
    assert res is not None
    assert len(get_fallback_history()) == 1


def test_dispatch_uses_autotuned_thresholds(tmp_path, monkeypatch):
    import parallax.backends.dispatch as dd
    from parallax.backends.autotune import save_cache
    from parallax.backends.cpuinfo import CpuInfo
    from parallax.backends.dispatch import candidate_backends

    monkeypatch.setattr(dd, "detect_cpu", lambda: CpuInfo(arch="x86_64", system="Linux", avx2=True))
    save_cache({"gemm": 4}, path=tmp_path / "autotune.json")
    cands = candidate_backends("gemm", n_elements=16, environ={"PARALLAX_CACHE_DIR": str(tmp_path)})
    assert cands[0] == "c-avx"


def test_dispatch_quarantines_repeatedly_failing_backend():
    from parallax.backends.dispatch import candidate_backends
    from parallax.backends.quarantine import get_default_quarantine

    registry.register("c", FaultyBackend("c", is_avail=True, supports_op=True))
    registry.register("numpy", CalculatorBackend("numpy", is_avail=True, supports_op=True))
    a = np.ones((2, 2), dtype=np.float32)
    for _ in range(3):
        dispatch("gemm", a, a, _environ={"PARALLAX_BACKEND": "c"})
    assert get_default_quarantine().is_quarantined("gemm", "c")
    assert "c" not in candidate_backends("gemm", n_elements=4, environ={"PARALLAX_BACKEND": "c"})


def test_dispatch_records_telemetry_and_normalizes_native_layout():
    from parallax.backends.telemetry import DispatchTelemetry

    seen_contiguous: list[bool] = []

    class ContigBackend(CalculatorBackend):
        def gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            seen_contiguous.append(bool(a.flags.c_contiguous and b.flags.c_contiguous))
            return a @ b

    DispatchTelemetry().reset()
    registry.register("c", ContigBackend("c", is_avail=True, supports_op=True))
    strided = np.arange(16, dtype=np.float32).reshape(2, 8)[:, ::4]
    assert not strided.flags.c_contiguous
    out = dispatch("gemm", strided, strided, _environ={"PARALLAX_BACKEND": "c"})
    assert seen_contiguous == [True]
    assert DispatchTelemetry().snapshot()[("gemm", "c")].count == 1
    np.testing.assert_allclose(out, strided @ strided)


def test_dispatch_parity_guard_falls_back_on_drift():
    class DriftingBackend(CalculatorBackend):
        def gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return (a @ b) + 1.0

    registry.register("c", DriftingBackend("c", is_avail=True, supports_op=True))
    registry.register("numpy", CalculatorBackend("numpy", is_avail=True, supports_op=True))
    a = np.ones((2, 2), dtype=np.float32)
    out = dispatch("gemm", a, a, _environ={"PARALLAX_BACKEND": "c", "PARALLAX_CHECK_PARITY": "1"})
    np.testing.assert_allclose(out, a @ a)
    assert any("parity check failed" in r.reason for r in get_fallback_history())
