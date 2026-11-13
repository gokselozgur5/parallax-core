"""Unified op dispatch with policy, environment overrides, and fallback chain."""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from parallax.backends import registry
from parallax.backends.base import Backend
from parallax.backends.cpuinfo import detect_cpu
from parallax.backends.env import backend_override
from parallax.backends.policy import backend_order

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FallbackRecord:
    """Record of a failed backend candidate and the fallback reason."""

    op: str
    backend: str
    reason: str
    exception: str | None = None


_fallback_history: list[FallbackRecord] = []
_fallback_lock = threading.Lock()


def get_fallback_history() -> list[FallbackRecord]:
    """Return a snapshot copy of recorded fallback events."""
    with _fallback_lock:
        return list(_fallback_history)


def clear_fallback_history() -> None:
    """Clear the recorded fallback history."""
    with _fallback_lock:
        _fallback_history.clear()


def _record_fallback(op: str, backend: str, reason: str, exc: Exception | None = None) -> None:
    """Record a fallback event into global history."""
    exc_str = f"{type(exc).__name__}: {exc}" if exc is not None else None
    record = FallbackRecord(op=op, backend=backend, reason=reason, exception=exc_str)
    with _fallback_lock:
        _fallback_history.append(record)
    _LOGGER.debug("Dispatch fallback for %s on %s: %s", op, backend, reason)


_scope_local = threading.local()


def set_scope_backend(backend: str | None) -> None:
    """Set the active backend override for the current thread."""
    _scope_local.backend = backend


def get_scope_backend() -> str | None:
    """Get the active backend override for the current thread."""
    return getattr(_scope_local, "backend", None)


def _resolve_backend(name: str) -> Backend | None:
    """Resolve backend name from registry or known fallbacks."""
    try:
        b = registry.get(name)
        if isinstance(b, Backend):
            return b
    except KeyError:
        pass
    if name in ("c-neon", "c-avx", "c"):
        try:
            return registry.get("c")
        except KeyError:
            return None
    return None
