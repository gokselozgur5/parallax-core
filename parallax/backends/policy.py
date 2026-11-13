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


def native_order(cpu: CpuLike) -> tuple[str, ...]:
    """Return the preferred backends for a CPU, always ending with NumPy."""
    arch = cpu.arch.strip().lower()
    arch = _ARCH_ALIASES.get(arch, arch)
    if arch == "arm64" and cpu.system == "Darwin":
        return ("metal", "c-neon", NUMPY)
    if arch == "arm64" and cpu.neon:
        return ("c-neon", NUMPY)
    if arch == "x86_64" and cpu.avx2:
        return ("c-avx", "rust", NUMPY)
    if arch == "x86_64":
        return ("rust", NUMPY)
    return (NUMPY,)
