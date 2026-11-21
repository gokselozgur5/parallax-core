"""Tests for resilient dispatch fallback chain."""

from __future__ import annotations

from typing import Any
import numpy as np
import pytest

from parallax.core.dispatch_chain import DispatchChain


class DummyBackend:
    name = "dummy"

    def available(self) -> bool:
        return True

    def supports(self, op: str, dtype: Any = None, shape: tuple[int, ...] = ()) -> bool:
        return op == "gemm"

    def gemm(self, a: int, b: int) -> int:
        return a + b


def test_dispatch_chain_success_primary():
    backend = DummyBackend()
    chain = DispatchChain(registry={"dummy": backend})
    res = chain.execute("gemm", 2, 3, candidate_override=["dummy"])
    assert res == 5
    assert chain.get_history() == []


def test_dispatch_chain_fallback_on_unsupported_and_error():
    class FaultyBackend(DummyBackend):
        def gemm(self, a: int, b: int) -> int:
            raise RuntimeError("engine fault")

    faulty = FaultyBackend()
    fallback = DummyBackend()
    chain = DispatchChain(registry={"faulty": faulty, "fb": fallback})
    res = chain.execute("gemm", 10, 20, candidate_override=["faulty", "fb"])
    assert res == 30
    hist = chain.get_history()
    assert len(hist) == 1
    assert hist[0].backend == "faulty"
    assert "engine fault" in (hist[0].exception or "")
    chain.clear_history()
    assert chain.get_history() == []


def test_dispatch_chain_all_backends_fail_raises():
    chain = DispatchChain(registry={})
    with pytest.raises(RuntimeError, match="All backends in fallback chain failed"):
        chain.execute("gemm", 1, 2, candidate_override=["missing_a", "missing_b"])


def test_dispatch_chain_module_helpers_and_custom_policy():
    from parallax.core.dispatch import DispatchPolicy
    from parallax.core.dispatch_chain import (
        clear_fallback_history,
        dispatch_chain,
        get_fallback_history,
    )

    clear_fallback_history()
    assert get_fallback_history() == []
    policy = DispatchPolicy(thresholds={"gemm": 100})
    chain = DispatchChain(policy=policy, registry={})
    assert chain.get_candidates("gemm", 50) == ["numpy"]
    with pytest.raises(RuntimeError):
        dispatch_chain("unknown_op", 1, candidate_override=["nonexistent"])
    assert len(get_fallback_history()) == 1
    clear_fallback_history()


def test_dispatch_chain_inspect_inputs_and_numpy_array():
    class ShapeBackend(DummyBackend):
        def gemm(self, a: Any, b: Any) -> Any:
            return (a.shape, a.dtype)

    backend = ShapeBackend()
    chain = DispatchChain(registry={"shape_be": backend})
    arr = np.zeros((4, 4), dtype=np.float32)
    res = chain.execute("gemm", arr, arr, candidate_override=["shape_be"])
    assert res == ((4, 4), np.dtype(np.float32))
