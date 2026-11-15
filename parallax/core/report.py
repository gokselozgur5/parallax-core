"""Markdown backend availability report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class BackendReportRow:
    """Row metadata representing backend availability and capabilities."""

    name: str
    available: bool
    isa_flags: str
    parity: str
    capabilities: tuple[str, ...] = ()


def format_isa_flags(info: Any = None) -> str:
    """Format detected or provided CPU ISA flags into a comma-separated list."""
    if info is None:
        try:
            from parallax.backends import cpuinfo
            info = cpuinfo.detect_cpu()
        except Exception:
            info = None
    flags: list[str] = []
    if getattr(info, "neon", False):
        flags.append("NEON")
    if getattr(info, "avx2", False):
        flags.append("AVX2")
    if getattr(info, "avx512f", False):
        flags.append("AVX-512")
    if getattr(info, "cuda", False):
        flags.append("CUDA")
    return ", ".join(flags) if flags else "none"


def _check_backend_available(backend: Any) -> bool:
    if isinstance(backend, bool):
        return backend
    if hasattr(backend, "available") and callable(backend.available):
        try:
            return bool(backend.available())
        except Exception:
            return False
    return True


def _collect_backend_items(override: dict[str, Any] | None) -> list[tuple[str, Any]]:
    if override is not None:
        return sorted(override.items(), key=lambda x: str(x[0]))
    try:
        from parallax.backends import registry
        names = registry.list_backends()
        return sorted([(n, registry.get(n)) for n in names], key=lambda x: str(x[0]))
    except Exception:
        return []


COMMON_OPS: tuple[str, ...] = ("gemm", "matmul", "softmax", "rmsnorm", "attention")


def _detect_capabilities(backend: Any, override: Sequence[str] | None = None) -> tuple[str, ...]:
    if override is not None:
        return tuple(override)
    if hasattr(backend, "capabilities"):
        caps = backend.capabilities
        return tuple(caps() if callable(caps) else caps)
    if hasattr(backend, "supports") and callable(backend.supports):
        detected: list[str] = []
        for op in COMMON_OPS:
            try:
                if backend.supports(op):
                    detected.append(op)
            except Exception:
                pass
        return tuple(detected)
    return ()
