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


class DispatchChain:
    """Resilient fallback execution chain across candidate backends."""

    def __init__(
        self,
        policy: DispatchPolicy | None = None,
        registry: Any | None = None,
    ) -> None:
        self.policy = policy if policy is not None else DispatchPolicy()
        self.registry = registry
        self._history: list[FallbackRecord] = []

    def get_history(self) -> list[FallbackRecord]:
        """Return shallow copy of recorded fallback events."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear recorded fallback history."""
        self._history.clear()

    def _record(self, op: str, backend: str, reason: str, exc: Exception | None = None) -> None:
        exc_str = f"{type(exc).__name__}: {exc}" if exc is not None else None
        self._history.append(FallbackRecord(op=op, backend=backend, reason=reason, exception=exc_str))
        _LOGGER.debug("Dispatch fallback for %s on %s: %s", op, backend, reason)

    def get_candidates(
        self,
        op: str,
        size_or_shape: int | Sequence[int] = 0,
        platform: PlatformInfo | None = None,
    ) -> list[str]:
        return self.policy.candidate_backends(op, size_or_shape, platform=platform)

    def execute(
        self,
        op: str,
        *args: Any,
        platform: PlatformInfo | None = None,
        candidate_override: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        n_elems, dtype, shape = _inspect_inputs(args)
        if candidate_override is not None:
            candidates = list(candidate_override)
        else:
            candidates = self.get_candidates(op, n_elems, platform=platform)
        return self._run_chain(op, candidates, dtype, shape, args, kwargs)

    def _run_chain(
        self,
        op: str,
        candidates: list[str],
        dtype: Any,
        shape: tuple[int, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        last_exc: Exception | None = None
        for name in candidates:
            backend = _resolve_backend(self.registry, name)
            if backend is None or (hasattr(backend, "available") and not backend.available()):
                self._record(op, name, "backend unavailable or unresolved")
                continue
            if hasattr(backend, "supports") and not backend.supports(op, dtype, shape):
                self._record(op, name, "backend does not support op")
                continue
            fn = getattr(backend, op, None)
            if not callable(fn):
                self._record(op, name, f"backend lacks callable op {op}")
                continue
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                self._record(op, name, str(exc), exc=exc)
        raise RuntimeError(f"All backends in fallback chain failed for op {op!r}: {candidates}") from last_exc

    def dispatch(self, op: str, *args: Any, **kwargs: Any) -> Any:
        """Alias for execute to provide drop-in dispatch interface."""
        return self.execute(op, *args, **kwargs)


_DEFAULT_CHAIN = DispatchChain()


def get_fallback_history() -> list[FallbackRecord]:
    """Return fallback history recorded by the default chain."""
    return _DEFAULT_CHAIN.get_history()


def clear_fallback_history() -> None:
    """Clear fallback history recorded by the default chain."""
    _DEFAULT_CHAIN.clear_history()
