"""Platform-aware backend ordering for dispatch."""

from __future__ import annotations

from typing import Protocol

NUMPY = "numpy"


class CpuLike(Protocol):
    """Structural view of a CPU probe result."""

    @property
    def arch(self) -> str: ...

    @property
    def system(self) -> str: ...

    @property
    def neon(self) -> bool: ...

    @property
    def avx2(self) -> bool: ...
