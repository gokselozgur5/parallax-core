"""Tests for resilient dispatch fallback chain."""

from __future__ import annotations

from typing import Any
import pytest

from parallax.core.dispatch_chain import DispatchChain, FallbackRecord


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
