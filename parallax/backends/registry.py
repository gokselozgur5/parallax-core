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
