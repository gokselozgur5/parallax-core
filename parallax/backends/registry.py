"""Registry of compute backends ordered by priority."""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_backends: dict[str, tuple[Any, int]] = {}


def register(name: str, backend: Any, priority: int = 0) -> None:
    """Store a backend under a unique name with an optional priority."""
    with _lock:
        if name in _backends:
            raise ValueError(f"backend already registered: {name!r}")
        _backends[name] = (backend, priority)


def get(name: str) -> Any:
    """Return the backend registered under name."""
    with _lock:
        return _backends[name][0]


def unregister(name: str) -> None:
    """Remove the backend registered under name."""
    with _lock:
        del _backends[name]


def clear() -> None:
    """Remove every registered backend."""
    with _lock:
        _backends.clear()


def list_backends() -> list[str]:
    """Return names by descending priority, ties in registration order."""
    with _lock:
        items = list(_backends.items())
    items.sort(key=lambda item: -item[1][1])
    return [name for name, _ in items]
