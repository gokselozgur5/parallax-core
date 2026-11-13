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
