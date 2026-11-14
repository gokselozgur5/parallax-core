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
            self._backends[name] = (backend, priority)

    def get(self, name: str) -> Any:
        with self._lock:
            return self._backends[name][0]
