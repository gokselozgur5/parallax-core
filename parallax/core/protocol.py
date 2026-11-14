"""Backend protocol and op capability query mechanisms."""

from __future__ import annotations

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


