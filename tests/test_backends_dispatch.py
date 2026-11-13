from __future__ import annotations

from typing import Any
import numpy as np
import pytest

from parallax.backends import registry
from parallax.backends.base import Backend
from parallax.backends.dispatch import (
    FallbackRecord,
    clear_fallback_history,
    dispatch,
    get_fallback_history,
)


class MockBackend(Backend):
    def __init__(self, name: str, is_avail: bool = True, supports_op: bool = True) -> None:
        self.name = name
        self._avail = is_avail
        self._supports = supports_op

    def available(self) -> bool:
        return self._avail

    def supports(self, op: str, dtype: Any, shape: tuple[int, ...]) -> bool:
        return self._supports
