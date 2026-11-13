"""Backend protocol: a small contract shared by all compute backends."""

from __future__ import annotations

import abc
from typing import Any


class Backend(abc.ABC):
    """Contract every compute backend implements.

    Dispatch asks a backend whether it is usable on this machine
    (``available``) and whether it can run a given operation for a
    dtype and shape (``supports``) before calling into it.
    """

    name: str

    @abc.abstractmethod
    def available(self) -> bool:
        """Return True if this backend can run on the current machine."""

    @abc.abstractmethod
    def supports(self, op: str, dtype: Any, shape: tuple[int, ...]) -> bool:
        """Return True if ``op`` is supported for ``dtype`` and ``shape``."""


def check_backend_contract(backend: Backend) -> None:
    """Validate that ``backend`` honours the Backend contract.

    Raises TypeError or ValueError with a descriptive message.
    """
    name = getattr(backend, "name", None)
    if not isinstance(name, str):
        raise TypeError(f"backend name must be a str, got {type(name).__name__}")
    if not name:
        raise ValueError("backend name must be a non-empty string")
    avail = backend.available()
    if not isinstance(avail, bool):
        raise TypeError(f"available() must return bool, got {type(avail).__name__}")
    sample = backend.supports("gemm", "float32", (2, 2))
    if not isinstance(sample, bool):
        raise TypeError(f"supports() must return bool, got {type(sample).__name__}")
