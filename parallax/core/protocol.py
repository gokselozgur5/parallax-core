"""Backend protocol and op capability query mechanisms."""

from __future__ import annotations

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
