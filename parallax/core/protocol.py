"""Backend protocol and op capability query mechanisms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BackendProtocol(Protocol):
    """Protocol interface that compute backends must satisfy."""

    name: str

    def available(self) -> bool:
        """Return True if this backend is usable in current environment."""
        ...

    def supports(self, op: str, dtype: Any = None, shape: tuple[int, ...] = ()) -> bool:
        """Return True if the backend supports the operation."""
        ...


Backend = BackendProtocol


@dataclass(frozen=True)
class OpQuery:
    """Descriptor for an op capability query."""

    op: str
    dtype: Any = None
    shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.op, str) or not self.op.strip():
            raise ValueError("op must be a non-empty string")


def query_capability(
    backend: Any,
    query: OpQuery | str,
    dtype: Any = None,
    shape: tuple[int, ...] = (),
    check_available: bool = True,
) -> bool:
    """Query whether backend supports the given operation."""
    if check_available and hasattr(backend, "available") and not backend.available():
        return False
    if isinstance(query, str):
        query = OpQuery(op=query, dtype=dtype, shape=shape)
    if not hasattr(backend, "supports"):
        return False
    return bool(backend.supports(query.op, query.dtype, query.shape))


def query_backend_capabilities(
    backend: Any,
    ops: Sequence[str | OpQuery],
    check_available: bool = True,
) -> dict[str, bool]:
    """Query capabilities for multiple operations on a backend."""
    results: dict[str, bool] = {}
    for item in ops:
        op_name = item.op if isinstance(item, OpQuery) else item
        if op_name in results:
            raise ValueError(f"duplicate op in query: {op_name}")
        results[op_name] = query_capability(
            backend,
            item,
            check_available=check_available,
        )
    return results


def check_backend_protocol(backend: Any) -> None:
    """Validate that backend honours the BackendProtocol contract."""
    name = getattr(backend, "name", None)
    if not isinstance(name, str):
        raise TypeError(f"backend name must be a str, got {type(name).__name__}")
    if not name.strip():
        raise ValueError("backend name must be a non-empty string")
    avail_fn = getattr(backend, "available", None)
    if not callable(avail_fn):
        raise TypeError("backend must define callable available()")
    avail = avail_fn()
    if not isinstance(avail, bool):
        raise TypeError(f"available() must return bool, got {type(avail).__name__}")
    supports_fn = getattr(backend, "supports", None)
    if not callable(supports_fn):
        raise TypeError("backend must define callable supports()")
    sample = supports_fn("gemm", None, ())
    if not isinstance(sample, bool):
        raise TypeError(f"supports() must return bool, got {type(sample).__name__}")




