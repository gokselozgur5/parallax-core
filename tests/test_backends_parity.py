from __future__ import annotations

from typing import Any
import numpy as np
import pytest

from parallax.backends.base import Backend
from parallax.backends.numpy_backend import NumpyBackend
from parallax.backends.parity import (
    ParityError,
    check_op_parity,
    check_parity,
    parity_errors,
)
from parallax.backends import registry


class MockBackend(Backend):
    def __init__(self, name: str, offset: float = 0.0, available: bool = True):
        self.name = name
        self.offset = offset
        self._available = available

    def available(self) -> bool:
        return self._available
