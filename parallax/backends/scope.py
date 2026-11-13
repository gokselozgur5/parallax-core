"""Thread-local backend scope management."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Generator
from typing import Any

_local = threading.local()


def _get_stack() -> list[tuple[str | None, str | None]]:
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = []
        _local.stack = stack
    return stack


def _normalize_name(val: Any, label: str) -> str | None:
    if val is None:
        return None
    name = getattr(val, "name", val)
    if not isinstance(name, str):
        raise TypeError(f"{label} must be a str or None; got {type(val).__name__}")
    cleaned = name.strip().lower()
    if not cleaned:
        raise ValueError(f"{label} cannot be empty")
    return cleaned


@contextlib.contextmanager
def use_backend(
    backend: Any,
    op: str | None = None,
) -> Generator[str | None, None, None]:
    """Temporarily override active compute backend in current thread."""
    stack = _get_stack()
    norm_backend = _normalize_name(backend, "backend")
    norm_op = _normalize_name(op, "op")
    stack.append((norm_op, norm_backend))
    try:
        yield norm_backend
    finally:
        stack.pop()


def current_backend(op: str | None = None) -> str | None:
    """Return currently active scoped backend for op or global scope."""
    stack = _get_stack()
    if not stack:
        return None
    norm_op = _normalize_name(op, "op") if op is not None else None
    for entry_op, backend in reversed(stack):
        if norm_op is not None and entry_op == norm_op:
            return backend
        if entry_op is None:
            return backend
    return None


def clear_scopes() -> None:
    """Clear all active scopes for current thread."""
    _get_stack().clear()


scoped_backend = current_backend

__all__ = [
    "clear_scopes",
    "current_backend",
    "scoped_backend",
    "use_backend",
]
