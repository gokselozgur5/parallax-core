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
