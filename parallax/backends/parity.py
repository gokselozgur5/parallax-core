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


def _resolve_backend(target: str | Backend) -> Backend:
    if isinstance(target, Backend):
        return target
    try:
        registered = registry.get(target)
        if isinstance(registered, Backend):
            return registered
        if callable(registered):
            return registered()
    except KeyError:
        pass
    if target == "numpy":
        from parallax.backends.numpy_backend import NumpyBackend

        return NumpyBackend()
    raise ValueError(f"unknown backend: {target!r}")


def _infer_layout(args: Sequence[Any]) -> tuple[Any, tuple[int, ...]]:
    for arg in args:
        if isinstance(arg, np.ndarray):
            return arg.dtype, arg.shape
    return np.float32, (1,)


def check_op_parity(
    backend: Backend,
    op: str,
    ref_output: np.ndarray,
    *args: Any,
    threshold: float = DEFAULT_PARITY_THRESHOLD,
    **kwargs: Any,
) -> ParityResult:
    name = backend.name
    if not backend.available():
        return ParityResult(name, op, 0.0, True, "unavailable")
    dtype, shape = _infer_layout(args)
    if not backend.supports(op, dtype, shape):
        return ParityResult(name, op, 0.0, True, "unsupported")
    try:
        fn = getattr(backend, op, None)
        if fn is None or not callable(fn):
            return ParityResult(name, op, 0.0, False, "error", f"missing op {op}")
        out = fn(*args, **kwargs)
        if not isinstance(out, np.ndarray):
            return ParityResult(name, op, 0.0, False, "error", "not ndarray")
        err = float(np.max(np.abs(out - ref_output)))
        passed = bool(err <= threshold)
        status = "ok" if passed else "flagged"
        return ParityResult(name, op, err, passed, status)
    except Exception as exc:
        return ParityResult(name, op, 0.0, False, "error", str(exc))


def check_parity(
    op: str,
    *args: Any,
    backends: Sequence[Backend | str] | None = None,
    reference: Backend | str = "numpy",
    threshold: float = DEFAULT_PARITY_THRESHOLD,
    raise_on_error: bool = False,
    **kwargs: Any,
) -> ParityReport:
    ref_b = _resolve_backend(reference)
    ref_fn = getattr(ref_b, op, None)
    if ref_fn is None:
        raise AttributeError(f"reference backend lacks op: {op}")
    ref_out = ref_fn(*args, **kwargs)

    targets: list[Backend] = []
    if backends is None:
        for name in registry.list_backends():
            targets.append(_resolve_backend(name))
    else:
        for item in backends:
            targets.append(_resolve_backend(item))
    report = ParityReport()
    for b in targets:
        report[b.name] = check_op_parity(
            b, op, ref_out, *args, threshold=threshold, **kwargs
        )

    if raise_on_error and not report.all_passed:
        flagged = report.flagged_backends
        raise ParityError(f"parity check failed for backends: {flagged}")
    return report


def parity_errors(
    op: str,
    *args: Any,
    threshold: float = DEFAULT_PARITY_THRESHOLD,
    **kwargs: Any,
) -> dict[str, float]:
    report = check_parity(op, *args, threshold=threshold, **kwargs)
    return report.max_errors
