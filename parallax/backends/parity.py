from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np

from parallax.backends.base import Backend
from parallax.backends import registry

DEFAULT_PARITY_THRESHOLD: float = 1e-4


class ParityError(RuntimeError):
    """Raised when parity check exceeds threshold or fails."""


@dataclass(frozen=True)
class ParityResult:
    backend: str
    op: str
    max_abs_error: float
    passed: bool
    status: str
    error_message: str | None = None
