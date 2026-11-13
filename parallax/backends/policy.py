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


# Below these element counts native call overhead dominates, so NumPy runs.
MIN_ELEMENTS: dict[str, int] = {
    "gemm": 64 * 64,
    "softmax": 4096,
    "rmsnorm": 4096,
    "attention": 32 * 64,
}

_ARCH_ALIASES = {"aarch64": "arm64", "amd64": "x86_64"}
