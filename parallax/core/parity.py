"""Cross-backend numerical parity validation framework."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np

DEFAULT_PARITY_THRESHOLD: float = 1e-4

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


def _to_numpy(val: Any) -> np.ndarray:
    return np.asarray(getattr(val, "data", val))


def _resolve_backend(registry: Any, target: Any) -> Any:
    if not isinstance(target, str):
        return target
    if registry is not None:
        getter = getattr(registry, "get", None) or (registry.get if isinstance(registry, dict) else None)
        if getter is not None:
            try:
                b = getter(target)
                if b is not None:
                    return b() if callable(b) else b
            except KeyError:
                pass
    if target == "numpy":
        from parallax.backends.numpy_backend import NumpyBackend

        return NumpyBackend()
    raise ValueError(f"unknown backend: {target!r}")


class ParityChecker:
    """Validator of numerical parity across candidate backends."""

    def __init__(
        self,
        registry: Any = None,
        reference: Any = "numpy",
        default_threshold: float = DEFAULT_PARITY_THRESHOLD,
    ) -> None:
        self.registry = registry
        self.reference = reference
        self.default_threshold = default_threshold

    def compare(self, actual: Any, ref: Any, threshold: float) -> tuple[float, float, bool]:
        act_arr, ref_arr = _to_numpy(actual), _to_numpy(ref)
        diff = np.abs(act_arr - ref_arr)
        abs_err = float(np.max(diff)) if diff.size > 0 else 0.0
        rel_denom = np.abs(ref_arr) + 1e-12
        rel_err = float(np.max(diff / rel_denom)) if diff.size > 0 else 0.0
        return abs_err, rel_err, bool(abs_err <= threshold)

    def check_op(
        self,
        backend: Any,
        op: str,
        ref_out: Any,
        *args: Any,
        threshold: float | None = None,
        **kwargs: Any,
    ) -> ParityResult:
        tol = self.default_threshold if threshold is None else threshold
        name = getattr(backend, "name", str(backend))
        if hasattr(backend, "available") and not backend.available():
            return ParityResult(name, op, 0.0, True, "unavailable")
        fn = getattr(backend, op, None)
        if not callable(fn):
            return ParityResult(name, op, 0.0, False, "error", f"missing op {op}")
        out = fn(*args, **kwargs)
        abs_err, rel_err, passed = self.compare(out, ref_out, tol)
        return ParityResult(name, op, abs_err, passed, "ok" if passed else "flagged", rel_err)



