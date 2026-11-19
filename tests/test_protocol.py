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


def test_op_query_attributes_and_validation():
    from parallax.core.protocol import OpQuery

    q = OpQuery("gemm", dtype="float32", shape=(2, 2))
    assert q.op == "gemm"
    assert q.dtype == "float32"
    assert q.shape == (2, 2)
    with pytest.raises(ValueError):
        OpQuery("")


def test_query_capability_mechanism():
    from parallax.core.protocol import OpQuery, query_capability

    class MockBackend:
        name = "mock"

        def available(self) -> bool:
            return True

        def supports(self, op: str, dtype: Any = None, shape: tuple[int, ...] = ()) -> bool:
            return op == "gemm" and shape == (2, 2)

    b = MockBackend()
    assert query_capability(b, "gemm", shape=(2, 2))
    assert not query_capability(b, "gemm", shape=(3, 3))
    assert query_capability(b, OpQuery("gemm", shape=(2, 2)))
    assert not query_capability(b, OpQuery("softmax"))


def test_query_backend_capabilities_batch():
    from parallax.core.protocol import query_backend_capabilities

    class MultiOpBackend:
        name = "multi"

        def available(self) -> bool:
            return True

        def supports(self, op: str, dtype: Any = None, shape: tuple[int, ...] = ()) -> bool:
            return op in ("gemm", "rmsnorm")

    b = MultiOpBackend()
    res = query_backend_capabilities(b, ["gemm", "softmax", "rmsnorm"])
    assert res == {"gemm": True, "softmax": False, "rmsnorm": True}


def test_check_backend_protocol_validation():
    from parallax.core.protocol import check_backend_protocol

    class ValidBackend:
        name = "valid"

        def available(self) -> bool:
            return True

        def supports(self, op: str, dtype: Any = None, shape: tuple[int, ...] = ()) -> bool:
            return True

    check_backend_protocol(ValidBackend())

    class InvalidName:
        name = ""

        def available(self) -> bool:
            return True

        def supports(self, op: str, dtype: Any = None, shape: tuple[int, ...] = ()) -> bool:
            return True

    with pytest.raises(ValueError):
        check_backend_protocol(InvalidName())


def test_numpy_backend_satisfies_protocol():
    from parallax.backends.numpy_backend import NumpyBackend
    from parallax.core.protocol import BackendProtocol, check_backend_protocol, query_capability

    nb = NumpyBackend()
    assert isinstance(nb, BackendProtocol)
    check_backend_protocol(nb)
    assert query_capability(nb, "gemm")
    assert not query_capability(nb, "unknown_op")


def test_check_backend_protocol_invalid_types():
    from parallax.core.protocol import check_backend_protocol

    class BadNameType:
        name = 123
        def available(self) -> bool: return True
        def supports(self, op: str, dtype: Any = None, shape: tuple[int, ...] = ()) -> bool: return True

    class BadAvailType:
        name = "test"
        def available(self) -> int: return 1  # type: ignore[return-value]
        def supports(self, op: str, dtype: Any = None, shape: tuple[int, ...] = ()) -> bool: return True

    class BadSupports:
        name = "test"
        def available(self) -> bool: return True
        def supports(self, op: str, dtype: Any = None, shape: tuple[int, ...] = ()) -> None: return None  # type: ignore[return-value]

    with pytest.raises(TypeError): check_backend_protocol(BadNameType())
    with pytest.raises(TypeError): check_backend_protocol(BadAvailType())
    with pytest.raises(TypeError): check_backend_protocol(BadSupports())








def test_query_backend_capabilities_duplicate_op():
    from parallax.core.protocol import OpQuery, query_backend_capabilities

    class ShapeBackend:
        name = "shape"
        def available(self) -> bool: return True
        def supports(self, op: str, dtype: Any = None, shape: tuple[int, ...] = ()) -> bool:
            return op == "gemm" and shape == (2, 2)

    b = ShapeBackend()
    queries = [OpQuery("gemm", shape=(2, 2)), OpQuery("gemm", shape=(3, 3))]
    with pytest.raises(ValueError, match="duplicate op"):
        query_backend_capabilities(b, queries)
