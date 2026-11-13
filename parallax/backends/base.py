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
