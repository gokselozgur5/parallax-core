"""Tests for backend protocol and op capability query mechanisms."""

from __future__ import annotations

from typing import Any
import pytest


def test_backend_protocol_check():
    from parallax.core.protocol import Backend, BackendProtocol

    class DummyBackend:
        name = "dummy"

        def available(self) -> bool:
            return True

        def supports(self, op: str, dtype: Any = None, shape: tuple[int, ...] = ()) -> bool:
            return op == "gemm"

    backend = DummyBackend()
    assert isinstance(backend, BackendProtocol)
    assert isinstance(backend, Backend)
