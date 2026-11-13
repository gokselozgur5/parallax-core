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


class ParityReport(dict[str, ParityResult]):
    """Mapping from backend name to its ParityResult."""

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.values())

    @property
    def max_errors(self) -> dict[str, float]:
        return {k: v.max_abs_error for k, v in self.items() if v.status == "ok"}

    @property
    def flagged_backends(self) -> list[str]:
        return [k for k, v in self.items() if not v.passed]
