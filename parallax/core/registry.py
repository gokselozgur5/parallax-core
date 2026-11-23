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

    def unregister(self, name: str) -> None:
        with self._lock:
            if name not in self._backends:
                raise KeyError(f"backend not found: {name!r}")
            del self._backends[name]

    def clear(self) -> None:
        with self._lock:
            self._backends.clear()

    def list_backends(self) -> list[str]:
        with self._lock:
            items = list(self._backends.items())
        items.sort(key=lambda item: -item[1][1])
        return [name for name, _ in items]

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._backends

    def __len__(self) -> int:
        with self._lock:
            return len(self._backends)

    def __getitem__(self, name: str) -> Any:
        return self.get(name)

    def __delitem__(self, name: str) -> None:
        self.unregister(name)


class _SynchronizedDefaultRegistry(BackendRegistry):
    """Default registry instance synchronized with parallax.backends.registry."""

    def register(self, name: str, backend: Any, priority: int = 0) -> None:
        super().register(name, backend, priority=priority)
        from parallax.backends import registry as b_reg

        try:
            b_reg.register(name, backend, priority=priority)
        except ValueError:
            pass

    def unregister(self, name: str) -> None:
        super().unregister(name)
        from parallax.backends import registry as b_reg

        try:
            b_reg.unregister(name)
        except KeyError:
            pass

    def clear(self) -> None:
        super().clear()
        from parallax.backends import registry as b_reg

        b_reg.clear()


_default_registry = _SynchronizedDefaultRegistry()
register = _default_registry.register
get = _default_registry.get
unregister = _default_registry.unregister
clear = _default_registry.clear
list_backends = _default_registry.list_backends


__all__ = [
    "BackendRegistry",
    "clear",
    "get",
    "list_backends",
    "register",
    "unregister",
]
