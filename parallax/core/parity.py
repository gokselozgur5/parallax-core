"""Cross-backend numerical parity validation framework."""
from __future__ import annotations
from dataclasses import dataclass

class ParityError(RuntimeError):
    """Raised when numerical parity validation fails."""

@dataclass(frozen=True)
class ParityResult:
    backend: str
    op: str
    max_abs_error: float
    passed: bool
    status: str
    max_rel_error: float = 0.0
    error_message: str | None = None

class ParityReport(dict[str, ParityResult]):
    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.values())

    @property
    def flagged_backends(self) -> list[str]:
        return [k for k, v in self.items() if not v.passed]

    @property
    def max_errors(self) -> dict[str, float]:
        return {k: v.max_abs_error for k, v in self.items() if v.status == "ok"}
