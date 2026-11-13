from __future__ import annotations

import threading
from typing import Optional


class BackendQuarantine:
    """Tracks consecutive op failures and quarantines failing backends."""

    def __init__(self, failure_threshold: int = 3) -> None:
        if failure_threshold < 1:
            raise ValueError(f"threshold must be >= 1, got {failure_threshold}")
        self._threshold = failure_threshold
        self._lock = threading.Lock()
        self._failures: dict[tuple[str, str], int] = {}
        self._quarantined: set[tuple[str, str]] = set()

    @property
    def threshold(self) -> int:
        return self._threshold
