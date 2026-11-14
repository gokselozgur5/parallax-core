"""Resilient fallback execution chain across registered backends."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from parallax.core.dispatch import DispatchPolicy, PlatformInfo

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FallbackRecord:
    """Record of a failed backend candidate and fallback reason."""

    op: str
    backend: str
    reason: str
    exception: str | None = None


def _inspect_inputs(args: tuple[Any, ...]) -> tuple[int, Any, tuple[int, ...]]:
    if not args:
        return 0, None, ()
    first = args[0]
    return (
        int(getattr(first, "size", 0)),
        getattr(first, "dtype", None),
        tuple(getattr(first, "shape", ())),
    )


def _resolve_backend(registry: Any, name: str) -> Any:
    if registry is None:
        from parallax.backends import registry as default_registry

        registry = default_registry
    if hasattr(registry, "get"):
        try:
            return registry.get(name)
        except KeyError:
            return None
    if isinstance(registry, dict):
        return registry.get(name)
    return getattr(registry, name, None)
