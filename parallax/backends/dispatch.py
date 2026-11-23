"""Unified op dispatch with policy, environment overrides, and fallback chain."""

from __future__ import annotations

import functools
import logging
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from parallax.backends import registry
from parallax.backends.autotune import get_default_cache_path, load_cache
from parallax.backends.base import Backend
from parallax.backends.cpuinfo import detect_cpu as _detect_cpu_raw
from parallax.backends.env import backend_override
from parallax.backends.policy import backend_order
from parallax.backends.quarantine import get_default_quarantine
from parallax.backends.scope import current_backend

_LOGGER = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def detect_cpu():
    """Cached wrapper around host CPU feature detection for dispatch."""
    return _detect_cpu_raw()


@dataclass(frozen=True)
class FallbackRecord:
    """Record of a failed backend candidate and the fallback reason."""

    op: str
    backend: str
    reason: str
    exception: str | None = None


_MAX_FALLBACK_HISTORY = 1000
_fallback_history: deque[FallbackRecord] = deque(maxlen=_MAX_FALLBACK_HISTORY)
_fallback_lock = threading.Lock()


def get_fallback_history() -> list[FallbackRecord]:
    """Return a snapshot copy of recorded fallback events."""
    with _fallback_lock:
        return list(_fallback_history)


def clear_fallback_history() -> None:
    """Clear the recorded fallback history and reset quarantine state."""
    with _fallback_lock:
        _fallback_history.clear()
    get_default_quarantine().reset()


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
    if name == "numpy":
        from parallax.backends.numpy_backend import NumpyBackend

        return NumpyBackend()
    return None


def _inspect_inputs(args: tuple[Any, ...]) -> tuple[int, Any, tuple[int, ...]]:
    if not args:
        return 0, None, ()
    first = args[0]
    return (
        int(getattr(first, "size", 0)),
        getattr(first, "dtype", None),
        tuple(getattr(first, "shape", ())),
    )


_autotune_state: tuple[str, int, dict[str, int] | None] | None = None


def _active_thresholds(environ: Mapping[str, str] | None = None) -> dict[str, int] | None:
    global _autotune_state
    if environ is not None and "PARALLAX_CACHE_DIR" in environ:
        path = Path(environ["PARALLAX_CACHE_DIR"]) / "autotune.json"
    else:
        path = get_default_cache_path()
    key = str(path)
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return None
    if _autotune_state is not None and _autotune_state[0] == key and _autotune_state[1] == mtime:
        return _autotune_state[2]
    try:
        data = load_cache(path)
    except Exception:
        data = None
    _autotune_state = (key, mtime, data)
    return data


def candidate_backends(
    op: str, n_elements: int = 0, environ: Mapping[str, str] | None = None
) -> list[str]:
    c: list[str] = []
    sb = current_backend(op) or get_scope_backend()
    if sb:
        c.append(sb)
    ov = backend_override(op, environ=environ)
    if ov and ov not in c:
        c.append(ov)
    try:
        for name in backend_order(op, n_elements, detect_cpu(), thresholds=_active_thresholds(environ)):
            if name not in c:
                c.append(name)
    except (ValueError, KeyError):
        if "numpy" not in c:
            c.append("numpy")
    quarantine = get_default_quarantine()
    return [name for name in c if name == "numpy" or not quarantine.is_quarantined(op, name)]


def dispatch(op: str, *args: Any, **kwargs: Any) -> Any:
    env = kwargs.pop("_environ", None)
    n_elems, dtype, shape = _inspect_inputs(args)
    candidates = candidate_backends(op, n_elements=n_elems, environ=env)
    last_err: Exception | None = None
    for name in candidates:
        b = _resolve_backend(name)
        if b is None or not b.available() or not b.supports(op, dtype, shape):
            _record_fallback(op, name, "backend not available or unsupported")
            continue
        fn = getattr(b, op, None)
        if not callable(fn):
            _record_fallback(op, name, f"backend lacks op {op}")
            continue
        try:
            out = fn(*args, **kwargs)
            get_default_quarantine().record_success(op, name)
            return out
        except Exception as exc:
            last_err = exc
            if name != "numpy":
                get_default_quarantine().record_failure(op, name)
            _record_fallback(op, name, str(exc), exc=exc)
    msg = f"All backends failed for {op}: {candidates}"
    raise RuntimeError(msg) from last_err
