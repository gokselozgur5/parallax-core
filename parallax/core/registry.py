"""Backend registry prioritizing execution engines."""

from __future__ import annotations

import threading
from typing import Any


class BackendRegistry:
    """Registry prioritizing execution engines."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._backends: dict[str, tuple[Any, int]] = {}

    def register(self, name: str, backend: Any, priority: int = 0) -> None:
        with self._lock:
            if name in self._backends:
                raise ValueError(f"backend already registered: {name!r}")
            self._backends[name] = (backend, priority)

    def get(self, name: str) -> Any:
        with self._lock:
            if name not in self._backends:
                raise KeyError(f"backend not found: {name!r}")
            return self._backends[name][0]
