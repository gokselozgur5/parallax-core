"""Platform-aware backend ordering for dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

NUMPY = "numpy"
ENABLE_NAIVE_GEMM_VAR = "PARALLAX_ENABLE_NAIVE_GEMM"


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


def backend_order(
    op: str,
    n_elements: int,
    cpu: CpuLike,
    thresholds: Mapping[str, int] | None = None,
) -> tuple[str, ...]:
    """Return the ordered backends to try for ``op`` on ``n_elements`` items."""
    if op not in MIN_ELEMENTS:
        raise ValueError(f"unknown op {op!r}; expected one of {sorted(MIN_ELEMENTS)}")
    min_elems = thresholds[op] if thresholds is not None and op in thresholds else MIN_ELEMENTS[op]
    if n_elements < min_elems:
        return (NUMPY,)
    return native_order(cpu)
